#include "lance_arrow_compat.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace duckdb {

namespace {

// ---------------------------------------------------------------------------
// Per-type coercion rules. Add new rules here; call sites never change.
// ---------------------------------------------------------------------------
//
// A rule describes a single Arrow type DuckDB cannot consume. Each rule
// provides:
//   * `matches(format)`: does this rule apply to the given Arrow format string?
//   * `coerced_format`:   static storage Arrow format the schema is rewritten
//   to.
//   * `convert(src, dst, n)`: widen `n` elements from the on-disk buffer into
//                              a DuckDB-native buffer. `dst` is a new buffer
//                              of `coerced_element_size * n` bytes.
//   * `src_element_size`:  bytes per element on disk.
//   * `coerced_element_size`: bytes per element after coercion.

struct CoercionRule {
  bool (*matches)(const char *format);
  const char *coerced_format;
  size_t src_element_size;
  size_t coerced_element_size;
  void (*convert)(const void *src, void *dst, int64_t count);
};

// --- Float16 → Float32 ------------------------------------------------------

// IEEE 754 half-precision (binary16) → single-precision (binary32) widening.
// Matches lance-spark's Float16Utils.halfToFloat bit-for-bit so readers across
// the two engines see identical values.
inline float HalfToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h >> 15) & 0x1u;
  uint32_t exponent = static_cast<uint32_t>(h >> 10) & 0x1Fu;
  uint32_t mantissa = static_cast<uint32_t>(h) & 0x3FFu;
  uint32_t bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign << 31;
    } else {
      int e = 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        e--;
      }
      mantissa &= 0x3FFu;
      uint32_t float_exp = static_cast<uint32_t>(e + (127 - 15));
      bits = (sign << 31) | (float_exp << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
  } else {
    uint32_t float_exp = exponent + (127u - 15u);
    bits = (sign << 31) | (float_exp << 23) | (mantissa << 13);
  }

  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

bool MatchesFloat16(const char *format) {
  return format != nullptr && format[0] == 'e' && format[1] == '\0';
}

void ConvertFloat16ToFloat32(const void *src, void *dst, int64_t count) {
  const auto *in = static_cast<const uint16_t *>(src);
  auto *out = static_cast<float *>(dst);
  for (int64_t i = 0; i < count; i++) {
    out[i] = HalfToFloat(in[i]);
  }
}

constexpr char kFloat32FormatLiteral[] = "f";

constexpr CoercionRule kRules[] = {
    {MatchesFloat16, kFloat32FormatLiteral, sizeof(uint16_t), sizeof(float),
     ConvertFloat16ToFloat32},
};

const CoercionRule *FindRule(const char *format) {
  for (const auto &rule : kRules) {
    if (rule.matches(format)) {
      return &rule;
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Generic recursion (unchanged by the set of coercion rules).
// ---------------------------------------------------------------------------

bool SchemaNeedsCoercion(const ArrowSchema *schema) {
  if (!schema) {
    return false;
  }
  if (FindRule(schema->format)) {
    return true;
  }
  for (int64_t i = 0; i < schema->n_children; i++) {
    if (SchemaNeedsCoercion(schema->children[i])) {
      return true;
    }
  }
  return SchemaNeedsCoercion(schema->dictionary);
}

// --- Array buffer rewrite ---------------------------------------------------

struct ArrayOverride {
  struct Entry {
    ArrowArray *array;
    const void *original_buffer;
    void *new_buffer;
  };
  std::vector<Entry> entries;
  void (*original_release)(ArrowArray *) = nullptr;
  void *original_private_data = nullptr;
};

void ArrayReleaseWrapper(ArrowArray *array) {
  if (!array || !array->private_data) {
    return;
  }
  auto *state = static_cast<ArrayOverride *>(array->private_data);
  for (auto &e : state->entries) {
    e.array->buffers[1] = e.original_buffer;
    std::free(e.new_buffer);
  }
  auto *original_release = state->original_release;
  array->release = original_release;
  array->private_data = state->original_private_data;
  delete state;
  if (original_release) {
    original_release(array);
  }
}

void CoerceArrayRecursive(const ArrowSchema *schema, ArrowArray *array,
                          ArrayOverride &state) {
  if (!schema || !array) {
    return;
  }
  if (const auto *rule = FindRule(schema->format)) {
    if (array->n_buffers >= 2 && array->buffers != nullptr &&
        array->buffers[1] != nullptr) {
      int64_t total = array->length + array->offset;
      if (total > 0) {
        size_t out_bytes =
            static_cast<size_t>(total) * rule->coerced_element_size;
        void *new_buf = std::malloc(out_bytes);
        if (!new_buf) {
          throw std::bad_alloc();
        }
        rule->convert(array->buffers[1], new_buf, total);
        ArrayOverride::Entry entry;
        entry.array = array;
        entry.original_buffer = array->buffers[1];
        entry.new_buffer = new_buf;
        state.entries.push_back(entry);
        array->buffers[1] = new_buf;
      }
    }
  }
  int64_t n = std::min<int64_t>(schema->n_children, array->n_children);
  for (int64_t i = 0; i < n; i++) {
    CoerceArrayRecursive(schema->children[i], array->children[i], state);
  }
  if (schema->dictionary && array->dictionary) {
    CoerceArrayRecursive(schema->dictionary, array->dictionary, state);
  }
}

// --- Schema format rewrite --------------------------------------------------

struct SchemaOverride {
  struct Entry {
    ArrowSchema *schema;
    const char *original_format;
  };
  std::vector<Entry> entries;
  void (*original_release)(ArrowSchema *) = nullptr;
  void *original_private_data = nullptr;
};

void SchemaReleaseWrapper(ArrowSchema *schema) {
  if (!schema || !schema->private_data) {
    return;
  }
  auto *state = static_cast<SchemaOverride *>(schema->private_data);
  for (auto &e : state->entries) {
    e.schema->format = e.original_format;
  }
  auto *original_release = state->original_release;
  schema->release = original_release;
  schema->private_data = state->original_private_data;
  delete state;
  if (original_release) {
    original_release(schema);
  }
}

void CoerceSchemaRecursive(ArrowSchema *schema, SchemaOverride &state) {
  if (!schema) {
    return;
  }
  if (const auto *rule = FindRule(schema->format)) {
    SchemaOverride::Entry entry;
    entry.schema = schema;
    entry.original_format = schema->format;
    state.entries.push_back(entry);
    schema->format = rule->coerced_format;
  }
  for (int64_t i = 0; i < schema->n_children; i++) {
    CoerceSchemaRecursive(schema->children[i], state);
  }
  CoerceSchemaRecursive(schema->dictionary, state);
}

} // namespace

bool LanceArrowSchemaNeedsCoercion(const ArrowSchema *schema) {
  return SchemaNeedsCoercion(schema);
}

std::vector<std::string> LanceCoerceArrowSchemaForDuckDB(ArrowSchema *schema) {
  std::vector<std::string> coerced_top_level;
  if (!schema || !SchemaNeedsCoercion(schema)) {
    return coerced_top_level;
  }

  // Capture affected top-level column names BEFORE installing the release
  // wrapper, so the list reflects user-facing catalog columns regardless of
  // whether the coercion lives at the top or inside a nested type.
  for (int64_t i = 0; i < schema->n_children; i++) {
    auto *child = schema->children[i];
    if (SchemaNeedsCoercion(child) && child && child->name) {
      coerced_top_level.emplace_back(child->name);
    }
  }

  auto *state = new SchemaOverride();
  state->original_release = schema->release;
  state->original_private_data = schema->private_data;
  try {
    CoerceSchemaRecursive(schema, *state);
  } catch (...) {
    for (auto &e : state->entries) {
      e.schema->format = e.original_format;
    }
    delete state;
    throw;
  }
  schema->private_data = state;
  schema->release = SchemaReleaseWrapper;
  return coerced_top_level;
}

void LanceCoerceArrowArrayForDuckDB(const ArrowSchema *schema,
                                    ArrowArray *array) {
  if (!schema || !array || array->release == nullptr) {
    return;
  }
  if (!SchemaNeedsCoercion(schema)) {
    return;
  }
  auto *state = new ArrayOverride();
  state->original_release = array->release;
  state->original_private_data = array->private_data;
  try {
    CoerceArrayRecursive(schema, array, *state);
  } catch (...) {
    for (auto &e : state->entries) {
      e.array->buffers[1] = e.original_buffer;
      std::free(e.new_buffer);
    }
    delete state;
    throw;
  }
  array->private_data = state;
  array->release = ArrayReleaseWrapper;
}

} // namespace duckdb
