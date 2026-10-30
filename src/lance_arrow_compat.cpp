#include "lance_arrow_compat.hpp"
#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/common/helper.hpp"

#include <memory>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {

bool LanceArrowFieldContainsBlob(const ArrowSchema &field) {
  if (field.metadata) {
    ArrowSchemaMetadata metadata(field.metadata);
    if (metadata.GetOption("lance-encoding:blob") == "true" ||
        metadata.GetOption("ARROW:extension:name") == "lance.blob.v2") {
      return true;
    }
  }
  if (field.dictionary && LanceArrowFieldContainsBlob(*field.dictionary)) {
    return true;
  }
  for (int64_t child_idx = 0; child_idx < field.n_children; child_idx++) {
    if (field.children && field.children[child_idx] &&
        LanceArrowFieldContainsBlob(*field.children[child_idx])) {
      return true;
    }
  }
  return false;
}

LanceArrowBatch::LanceArrowBatch(void *batch, const char *export_error) {
  std::unique_ptr<void, decltype(&lance_free_batch)> batch_owner(
      batch, lance_free_batch);
  array = make_shared_ptr<ArrowArrayWrapper>();
  array->arrow_array = {};
  schema.arrow_schema = {};
  if (lance_batch_to_arrow(batch, &array->arrow_array, &schema.arrow_schema) !=
      0) {
    throw IOException(string(export_error) + LanceFormatErrorSuffix());
  }
  batch_owner.reset();
  LanceCoerceArrowArrayForDuckDB(&schema.arrow_schema, &array->arrow_array);
}

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

constexpr char kArrowJsonExtension[] = "arrow.json";

bool HasArrowJsonExtension(const ArrowSchema *schema) {
  if (!schema || !schema->metadata) {
    return false;
  }
  ArrowSchemaMetadata metadata(schema->metadata);
  return metadata.GetOption(ArrowSchemaMetadata::ARROW_EXTENSION_NAME) ==
         kArrowJsonExtension;
}

bool ContainsArrowJsonField(const ArrowSchema *schema) {
  if (!schema) {
    return false;
  }
  if (HasArrowJsonExtension(schema)) {
    return true;
  }
  for (int64_t child_idx = 0; child_idx < schema->n_children; child_idx++) {
    if (schema->children &&
        ContainsArrowJsonField(schema->children[child_idx])) {
      return true;
    }
  }
  return ContainsArrowJsonField(schema->dictionary);
}

bool SchemaNeedsStorageCoercion(const ArrowSchema *schema) {
  if (!schema) {
    return false;
  }
  if (FindRule(schema->format)) {
    return true;
  }
  for (int64_t child_idx = 0; child_idx < schema->n_children; child_idx++) {
    if (SchemaNeedsStorageCoercion(schema->children[child_idx])) {
      return true;
    }
  }
  return SchemaNeedsStorageCoercion(schema->dictionary);
}

bool SchemaNeedsRewrite(const ArrowSchema *schema) {
  if (!schema) {
    return false;
  }
  if (FindRule(schema->format) || HasArrowJsonExtension(schema)) {
    return true;
  }
  for (int64_t child_idx = 0; child_idx < schema->n_children; child_idx++) {
    if (SchemaNeedsRewrite(schema->children[child_idx])) {
      return true;
    }
  }
  return SchemaNeedsRewrite(schema->dictionary);
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
        std::unique_ptr<void, decltype(&std::free)> new_buf(
            std::malloc(out_bytes), std::free);
        if (!new_buf) {
          throw std::bad_alloc();
        }
        rule->convert(array->buffers[1], new_buf.get(), total);
        ArrayOverride::Entry entry;
        entry.array = array;
        entry.original_buffer = array->buffers[1];
        entry.new_buffer = new_buf.get();
        state.entries.push_back(entry);
        array->buffers[1] = new_buf.release();
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

// --- Schema format and metadata rewrite ------------------------------------

struct SchemaOverride {
  struct Entry {
    ArrowSchema *schema;
    const char *original_format;
    const char *original_metadata;
    unsafe_unique_array<char> replacement_metadata;
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
    e.schema->metadata = e.original_metadata;
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
  const auto *rule = FindRule(schema->format);
  const bool strip_json_extension = HasArrowJsonExtension(schema);
  if (rule || strip_json_extension) {
    unsafe_unique_array<char> replacement_metadata;
    if (strip_json_extension) {
      ArrowSchemaMetadata metadata(schema->metadata);
      metadata.AddOption(ArrowSchemaMetadata::ARROW_EXTENSION_NAME, "");
      metadata.AddOption(ArrowSchemaMetadata::ARROW_METADATA_KEY, "");
      replacement_metadata = metadata.SerializeMetadata();
    }
    const char *replacement_ptr = replacement_metadata.get();
    SchemaOverride::Entry entry;
    entry.schema = schema;
    entry.original_format = schema->format;
    entry.original_metadata = schema->metadata;
    entry.replacement_metadata = std::move(replacement_metadata);
    state.entries.push_back(std::move(entry));
    if (rule) {
      schema->format = rule->coerced_format;
    }
    if (strip_json_extension) {
      schema->metadata = replacement_ptr;
    }
  }
  for (int64_t child_idx = 0; child_idx < schema->n_children; child_idx++) {
    CoerceSchemaRecursive(schema->children[child_idx], state);
  }
  CoerceSchemaRecursive(schema->dictionary, state);
}

} // namespace

bool LanceArrowSchemaNeedsCoercion(const ArrowSchema *schema) {
  return SchemaNeedsRewrite(schema);
}

LanceArrowSchemaCoercionResult
LanceCoerceArrowSchemaForDuckDB(ArrowSchema *schema) {
  LanceArrowSchemaCoercionResult result;
  if (!schema || !SchemaNeedsRewrite(schema)) {
    return result;
  }

  // Capture affected top-level column names before installing the release
  // wrapper. JSON remains a containment marker for the whole top-level field.
  for (int64_t child_idx = 0; child_idx < schema->n_children; child_idx++) {
    auto *child = schema->children[child_idx];
    if (!child || !child->name) {
      continue;
    }
    if (SchemaNeedsStorageCoercion(child)) {
      result.storage_coerced_columns.emplace_back(child->name);
    }
    if (ContainsArrowJsonField(child)) {
      result.json_columns.emplace_back(child->name);
    }
  }

  auto state = make_uniq<SchemaOverride>();
  state->original_release = schema->release;
  state->original_private_data = schema->private_data;
  try {
    CoerceSchemaRecursive(schema, *state);
  } catch (...) {
    for (auto &e : state->entries) {
      e.schema->format = e.original_format;
      e.schema->metadata = e.original_metadata;
    }
    throw;
  }
  schema->private_data = state.release();
  schema->release = SchemaReleaseWrapper;
  return result;
}

void LanceCoerceArrowArrayForDuckDB(const ArrowSchema *schema,
                                    ArrowArray *array) {
  if (!schema || !array || array->release == nullptr) {
    return;
  }
  if (!SchemaNeedsStorageCoercion(schema)) {
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

void LanceNormalizeArrowListFieldNames(ArrowSchema *schema) {
  if (!schema || schema->release == nullptr) {
    return;
  }
  static constexpr const char ARROW_LIST_ITEM_NAME[] = "item";
  if (schema->format &&
      (std::strcmp(schema->format, "+l") == 0 ||
       std::strcmp(schema->format, "+L") == 0) &&
      schema->n_children == 1 && schema->children && schema->children[0]) {
    // DuckDB assigns a string literal here, so overwriting the pointer with
    // another literal keeps the producer's release callback correct.
    schema->children[0]->name = ARROW_LIST_ITEM_NAME;
  }
  for (int64_t i = 0; i < schema->n_children; i++) {
    if (schema->children && schema->children[i]) {
      LanceNormalizeArrowListFieldNames(schema->children[i]);
    }
  }
}

} // namespace duckdb
