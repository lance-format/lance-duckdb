#include "lance_blob.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/cast/default_casts.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/planner/bound_statement.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_merge_into.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/planner_extension.hpp"

#include "lance_merge.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"

#include <algorithm>
#include <functional>
#include <limits>

namespace duckdb {

static constexpr const char *LANCE_BLOB_INPUT_TYPE_NAME = "LANCE_BLOB_INPUT";
static constexpr idx_t LANCE_BLOB_INGEST_BUDGET_FALLBACK =
    512ULL * 1024ULL * 1024ULL;

static LogicalType LanceBlobDescriptorType() {
  return LogicalType::STRUCT({{"uri", LogicalType::VARCHAR},
                              {"position", LogicalType::UBIGINT},
                              {"size", LogicalType::UBIGINT}});
}

static child_list_t<LogicalType> LanceBlobInputMembers() {
  child_list_t<LogicalType> members;
  members.emplace_back("bytes", LogicalType::BLOB);
  members.emplace_back("reference", LanceBlobDescriptorType());
  members.emplace_back("ingest", LanceBlobDescriptorType());
  return members;
}

LogicalType LanceBlobInputType() {
  return LogicalType::UNION(LanceBlobInputMembers())
      .WithAlias(LANCE_BLOB_INPUT_TYPE_NAME);
}

static bool IsBlobDescriptorType(const LogicalType &type) {
  if (type.id() != LogicalTypeId::STRUCT ||
      StructType::GetChildCount(type) != 3) {
    return false;
  }
  return StructType::GetChildName(type, 0).GetIdentifierName() == "uri" &&
         StructType::GetChildType(type, 0) == LogicalType::VARCHAR &&
         StructType::GetChildName(type, 1).GetIdentifierName() == "position" &&
         StructType::GetChildType(type, 1) == LogicalType::UBIGINT &&
         StructType::GetChildName(type, 2).GetIdentifierName() == "size" &&
         StructType::GetChildType(type, 2) == LogicalType::UBIGINT;
}

bool LanceIsBlobInput(const LogicalType &type) {
  if (type.id() != LogicalTypeId::UNION) {
    return false;
  }
  if (type == LanceBlobInputType()) {
    return true;
  }
  // DuckDB expression composition (UNION ALL, COALESCE, temporary tables, ...)
  // can drop the type alias. The member contract is the durable identity.
  if (UnionType::GetMemberCount(type) != 3) {
    return false;
  }
  return UnionType::GetMemberName(type, 0).GetIdentifierName() == "bytes" &&
         UnionType::GetMemberType(type, 0) == LogicalType::BLOB &&
         UnionType::GetMemberName(type, 1).GetIdentifierName() == "reference" &&
         IsBlobDescriptorType(UnionType::GetMemberType(type, 1)) &&
         UnionType::GetMemberName(type, 2).GetIdentifierName() == "ingest" &&
         IsBlobDescriptorType(UnionType::GetMemberType(type, 2));
}

// Require an absolute URI. Relative paths would otherwise be resolved against
// the query process's working directory, which is not part of the SQL value
// contract. Scheme support and URI normalization stay with Lance.
static void ValidateBlobUri(const string &uri, const char *function_name) {
  if (uri.empty()) {
    throw InvalidInputException("%s URI must not be empty", function_name);
  }
  auto scheme_end = uri.find("://");
  bool valid_scheme = scheme_end != string::npos && scheme_end > 0 &&
                      StringUtil::CharacterIsAlpha(uri[0]);
  for (idx_t i = 1; valid_scheme && i < scheme_end; i++) {
    auto c = uri[i];
    valid_scheme = StringUtil::CharacterIsAlphaNumeric(c) || c == '+' ||
                   c == '-' || c == '.';
  }
  if (!valid_scheme) {
    throw InvalidInputException(
        "%s requires an absolute URI with a scheme (e.g. file:///path or "
        "s3://bucket/key), got '%s'",
        function_name, uri);
  }
}

static string NormalizeBlobSourceUri(const string &uri) {
  const char *normalized = nullptr;
  if (lance_normalize_source_uri(uri.c_str(), &normalized) != 0) {
    throw InvalidInputException("Invalid source URI \"%s\"%s", uri,
                                LanceFormatErrorSuffix());
  }
  string result = normalized ? normalized : "";
  if (normalized) {
    lance_free_string(normalized);
  }
  return result;
}

static Value MakeBytesInput(const Value &bytes) {
  if (bytes.IsNull()) {
    return Value(LanceBlobInputType());
  }
  return Value::UNION(LanceBlobInputMembers(), 0,
                      bytes.DefaultCastAs(LogicalType::BLOB))
      .DefaultCastAs(LanceBlobInputType());
}

static Value MakeUriInput(LanceBlobKind kind, const string &uri, bool has_range,
                          uint64_t position, uint64_t size) {
  Value position_value = Value(LogicalType::UBIGINT);
  Value size_value = Value(LogicalType::UBIGINT);
  if (has_range) {
    position_value = Value::UBIGINT(position);
    size_value = Value::UBIGINT(size);
  }
  auto descriptor = Value::STRUCT({{"uri", Value(uri)},
                                   {"position", std::move(position_value)},
                                   {"size", std::move(size_value)}});
  return Value::UNION(LanceBlobInputMembers(), static_cast<uint8_t>(kind),
                      std::move(descriptor))
      .DefaultCastAs(LanceBlobInputType());
}

Value LanceMakeBytesBlobInputValue(const string &bytes) {
  // This value comes from an already-materialized binary buffer. Constructing
  // a VARCHAR first would reinterpret backslash escapes and reject non-UTF8
  // bytes during the cast to BLOB.
  return MakeBytesInput(Value::BLOB_RAW(bytes));
}

idx_t NormalizeIngestRowsInChunk(ClientContext &context, DataChunk &chunk) {
  vector<idx_t> blob_columns;
  for (idx_t column_idx = 0; column_idx < chunk.ColumnCount(); column_idx++) {
    if (LanceIsBlobInput(chunk.data[column_idx].GetType())) {
      blob_columns.push_back(column_idx);
    }
  }
  if (blob_columns.empty()) {
    return 0;
  }

  LanceBatchSourceOptions source_options;
  LanceCollectIngestSourceOptions(context, chunk, source_options);
  if (source_options.uris.empty()) {
    return 0;
  }
  LanceBlobSourceOptionArrays option_arrays;
  BuildLanceBlobSourceOptionArrays(source_options, option_arrays);

  auto remaining_budget = LanceBlobIngestBatchBudget(context);
  idx_t materialized = 0;
  for (auto column_idx : blob_columns) {
    vector<pair<idx_t, string>> replacements;
    for (idx_t row = 0; row < chunk.size(); row++) {
      LanceBlobInput input;
      string error;
      if (!LanceNormalizeBlobInput(chunk.GetValue(column_idx, row), input,
                                   error)) {
        continue;
      }
      if (input.kind != LanceBlobKind::INGEST) {
        continue;
      }
      uint64_t position = input.position;
      uint64_t size = input.size;
      const uint8_t *position_ptr =
          input.has_range ? reinterpret_cast<const uint8_t *>(&position)
                          : nullptr;
      const uint8_t *size_ptr =
          input.has_range ? reinterpret_cast<const uint8_t *>(&size) : nullptr;
      const uint8_t *bytes = nullptr;
      size_t len = 0;
      auto rc = lance_blob_read_source(
          input.uri.c_str(), position_ptr, size_ptr, option_arrays.Uris(),
          option_arrays.KeyOffsets(), option_arrays.ValueOffsets(),
          option_arrays.Counts(), option_arrays.counts.size(),
          static_cast<uint64_t>(remaining_budget),
          LanceGetSessionHandle(context), &bytes, &len);
      if (rc != 0) {
        throw IOException("Failed to materialize Lance Blob ingest value" +
                          LanceFormatErrorSuffix());
      }
      string data(reinterpret_cast<const char *>(bytes), len);
      lance_free_blob_bytes(bytes, len);
      if (len > remaining_budget) {
        throw IOException(
            "Lance Blob ingest value exceeded the per-batch materialization "
            "budget");
      }
      remaining_budget -= len;
      replacements.emplace_back(row, std::move(data));
    }
    if (replacements.empty()) {
      continue;
    }
    // Materialize dictionary/selection-backed inputs before replacing the
    // union member. VectorStructBuffer::SetValue requires a flat vector.
    chunk.data[column_idx].Flatten();
    for (auto &replacement : replacements) {
      chunk.data[column_idx].SetValue(
          replacement.first, LanceMakeBytesBlobInputValue(replacement.second));
      materialized++;
    }
  }
  return materialized;
}

static void ValidateBlobRange(const Value &offset_value,
                              const Value &length_value,
                              const char *function_name, uint64_t &out_position,
                              uint64_t &out_size) {
  if (offset_value.IsNull() || length_value.IsNull()) {
    throw InvalidInputException(
        "%s range arguments must both be non-NULL when a range is supplied",
        function_name);
  }
  auto offset = offset_value.GetValue<int64_t>();
  auto length = length_value.GetValue<int64_t>();
  if (offset < 0) {
    throw InvalidInputException("%s offset must be >= 0, got %lld",
                                function_name, static_cast<long long>(offset));
  }
  if (length <= 0) {
    throw InvalidInputException("%s length must be > 0, got %lld",
                                function_name, static_cast<long long>(length));
  }
  auto position = static_cast<uint64_t>(offset);
  auto size = static_cast<uint64_t>(length);
  if (position > std::numeric_limits<uint64_t>::max() - size) {
    throw InvalidInputException(
        "%s offset + length overflows the supported external range",
        function_name);
  }
  out_position = position;
  out_size = size;
}

static void LanceBlobConstructor(DataChunk &args, ExpressionState &,
                                 Vector &result, LanceBlobKind kind) {
  const auto &function_name =
      kind == LanceBlobKind::INGEST ? "lance_blob_ingest" : "lance_blob_ref";
  const bool ranged = args.ColumnCount() == 3;
  for (idx_t row = 0; row < args.size(); row++) {
    auto uri_value = args.GetValue(0, row);
    if (uri_value.IsNull()) {
      result.SetValue(row, Value(LanceBlobInputType()));
      continue;
    }
    auto uri = uri_value.GetValue<string>();
    ValidateBlobUri(uri, function_name);
    bool has_range = false;
    uint64_t position = 0;
    uint64_t size = 0;
    if (ranged) {
      ValidateBlobRange(args.GetValue(1, row), args.GetValue(2, row),
                        function_name, position, size);
      has_range = true;
    }
    result.SetValue(row, MakeUriInput(kind, uri, has_range, position, size));
  }
}

static void LanceBlobRefFunction(DataChunk &args, ExpressionState &state,
                                 Vector &result) {
  LanceBlobConstructor(args, state, result, LanceBlobKind::REFERENCE);
}

static void LanceBlobIngestFunction(DataChunk &args, ExpressionState &state,
                                    Vector &result) {
  LanceBlobConstructor(args, state, result, LanceBlobKind::INGEST);
}

static bool LanceBlobBytesCast(Vector &source, Vector &result, idx_t count,
                               CastParameters &) {
  for (idx_t row = 0; row < count; row++) {
    auto value = source.GetValue(row);
    result.SetValue(row, MakeBytesInput(value));
  }
  return true;
}

bool LanceNormalizeBlobInput(const Value &value, LanceBlobInput &out,
                             string &error) {
  out = LanceBlobInput();
  if (value.IsNull()) {
    return true;
  }
  if (value.type().id() == LogicalTypeId::BLOB) {
    out.is_null = false;
    out.kind = LanceBlobKind::BYTES;
    out.bytes = StringValue::Get(value);
    return true;
  }
  if (!LanceIsBlobInput(value.type())) {
    error = "value is not a LANCE_BLOB_INPUT";
    return false;
  }
  auto tag = UnionValue::GetTag(value);
  const auto &member = UnionValue::GetValue(value);
  if (tag == static_cast<uint8_t>(LanceBlobKind::BYTES)) {
    if (member.IsNull()) {
      error = "non-NULL LANCE_BLOB_INPUT has NULL bytes";
      return false;
    }
    out.is_null = false;
    out.kind = LanceBlobKind::BYTES;
    out.bytes = StringValue::Get(member.DefaultCastAs(LogicalType::BLOB));
    return true;
  }
  if (tag != static_cast<uint8_t>(LanceBlobKind::REFERENCE) &&
      tag != static_cast<uint8_t>(LanceBlobKind::INGEST)) {
    error = "LANCE_BLOB_INPUT has an unknown variant";
    return false;
  }
  if (member.IsNull() || member.type().id() != LogicalTypeId::STRUCT) {
    error = "non-NULL LANCE_BLOB_INPUT has a NULL URI descriptor";
    return false;
  }
  const auto &children = StructValue::GetChildren(member);
  if (children.size() != 3 || children[0].IsNull()) {
    error = "LANCE_BLOB_INPUT URI descriptor is missing its uri";
    return false;
  }
  auto uri = children[0].GetValue<string>();
  if (uri.empty()) {
    error = "LANCE_BLOB_INPUT URI must not be empty";
    return false;
  }
  bool has_position = !children[1].IsNull();
  bool has_size = !children[2].IsNull();
  if (has_position != has_size) {
    error = "LANCE_BLOB_INPUT must set both position and size, or neither";
    return false;
  }
  out.is_null = false;
  out.kind = tag == static_cast<uint8_t>(LanceBlobKind::REFERENCE)
                 ? LanceBlobKind::REFERENCE
                 : LanceBlobKind::INGEST;
  out.uri = std::move(uri);
  if (has_position) {
    out.has_range = true;
    out.position = children[1].GetValue<uint64_t>();
    out.size = children[2].GetValue<uint64_t>();
    if (out.size == 0 ||
        out.position > std::numeric_limits<uint64_t>::max() - out.size) {
      error = "LANCE_BLOB_INPUT external range is invalid";
      return false;
    }
  }
  return true;
}

idx_t LanceBlobIngestBatchBudget(ClientContext &context) {
  auto max_memory = DBConfig::GetConfig(context).options.maximum_memory;
  if (max_memory == DConstants::INVALID_INDEX || max_memory == 0) {
    // DuckDB normally resolves its default before a query runs; this fallback
    // only guards callers that observe an unresolved config.
    return LANCE_BLOB_INGEST_BUDGET_FALLBACK;
  }
  // One eighth of DuckDB's memory limit is the per-batch ingest payload budget
  // shared by every blob column. Buffered batches awaiting vector-dimension
  // inference and the writer channel are separate bounded copies; this is not
  // an overall query or process memory bound.
  return std::max<idx_t>(max_memory / 8, 1);
}

void LanceCollectIngestSourceOptions(ClientContext &context, DataChunk &chunk,
                                     LanceBatchSourceOptions &out) {
  out = LanceBatchSourceOptions();
  unordered_set<string> seen;
  for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
    if (!LanceIsBlobInput(chunk.data[col].GetType())) {
      continue;
    }
    for (idx_t row = 0; row < chunk.size(); row++) {
      LanceBlobInput input;
      string error;
      auto value = chunk.data[col].GetValue(row);
      if (!LanceNormalizeBlobInput(value, input, error) || input.is_null ||
          input.kind != LanceBlobKind::INGEST) {
        continue;
      }
      if (!seen.insert(input.uri).second) {
        continue;
      }
      vector<string> keys;
      vector<string> values;
      // Admission and secret selection must use the same resource identity
      // that Lance resolves before opening the source object store.
      auto normalized_uri =
          NormalizeBlobSourceUri(LanceNormalizeS3Scheme(input.uri));
      // I/O happens in the sink, so the source URI is subject to DuckDB's
      // access policy exactly like any other file that the query would read.
      // Allowing the target dataset does not allow arbitrary sources.
      auto &config = DBConfig::GetConfig(context);
      if (!config.CanAccessFile(normalized_uri, FileType::FILE_TYPE_REGULAR,
                                ClientData::Get(context).file_opener.get())) {
        throw PermissionException(
            "Cannot access source URI \"%s\" - file system operations are "
            "disabled by configuration",
            normalized_uri);
      }
      LanceFillStorageOptionsFromSecrets(context, normalized_uri, keys, values);
      out.uris.push_back(std::move(input.uri));
      out.keys.push_back(std::move(keys));
      out.values.push_back(std::move(values));
    }
  }
}

void BuildLanceBlobSourceOptionArrays(
    const LanceBatchSourceOptions &source_options,
    LanceBlobSourceOptionArrays &out) {
  out = LanceBlobSourceOptionArrays();
  size_t total = 0;
  for (auto &keys : source_options.keys) {
    total += keys.size();
  }
  out.uris.reserve(source_options.uris.size());
  out.counts.reserve(source_options.uris.size());
  out.key_offsets.reserve(source_options.uris.size());
  out.value_offsets.reserve(source_options.uris.size());
  out.all_keys.reserve(total);
  out.all_values.reserve(total);
  for (auto &uri : source_options.uris) {
    out.uris.push_back(uri.c_str());
  }
  for (idx_t i = 0; i < source_options.uris.size(); i++) {
    auto &keys = source_options.keys[i];
    auto &values = source_options.values[i];
    if (keys.size() != values.size()) {
      throw InternalException(
          "Lance batch source option keys/values size mismatch");
    }
    out.counts.push_back(keys.size());
    for (idx_t j = 0; j < keys.size(); j++) {
      out.all_keys.push_back(keys[j].c_str());
      out.all_values.push_back(values[j].c_str());
    }
  }
  // Per-URI offsets are only meaningful once all option pointers are final.
  idx_t offset = 0;
  for (auto count : out.counts) {
    out.key_offsets.push_back(count == 0 ? nullptr
                                         : out.all_keys.data() + offset);
    out.value_offsets.push_back(count == 0 ? nullptr
                                           : out.all_values.data() + offset);
    offset += count;
  }
}

int32_t LanceWriteBlobBatch(void *writer, ArrowArray *array,
                            const LanceBatchSourceOptions &source_options) {
  LanceBlobSourceOptionArrays arrays;
  BuildLanceBlobSourceOptionArrays(source_options, arrays);
  return lance_writer_write_batch_with_source_options(
      writer, array, arrays.Uris(), arrays.KeyOffsets(), arrays.ValueOffsets(),
      arrays.Counts(), arrays.counts.size());
}

vector<uint8_t> LanceBlobFlagsForTypes(const vector<LogicalType> &types) {
  std::function<bool(const LogicalType &)> contains_blob_input =
      [&](const LogicalType &type) {
        if (LanceIsBlobInput(type)) {
          return true;
        }
        switch (type.id()) {
        case LogicalTypeId::LIST:
          return contains_blob_input(ListType::GetChildType(type));
        case LogicalTypeId::STRUCT: {
          for (idx_t i = 0; i < StructType::GetChildCount(type); i++) {
            if (contains_blob_input(StructType::GetChildType(type, i))) {
              return true;
            }
          }
          return false;
        }
        case LogicalTypeId::MAP:
          return contains_blob_input(MapType::KeyType(type)) ||
                 contains_blob_input(MapType::ValueType(type));
        case LogicalTypeId::ARRAY:
          return contains_blob_input(ArrayType::GetChildType(type));
        default:
          return false;
        }
      };
  vector<uint8_t> flags;
  flags.reserve(types.size());
  for (auto &type : types) {
    flags.push_back(
        type.id() == LogicalTypeId::BLOB || contains_blob_input(type) ? 1 : 0);
  }
  return flags;
}

// Remove only the bound cast the INSERT binder adds on top of an input-typed
// blob projection. User-written casts are never rewritten: removing a cast
// whose child is not already the input type would change SQL semantics.
static void NormalizeLanceInsertBlobCasts(LogicalOperator &op) {
  for (auto &child : op.children) {
    if (child) {
      NormalizeLanceInsertBlobCasts(*child);
    }
  }
  if (op.type != LogicalOperatorType::LOGICAL_INSERT) {
    return;
  }
  auto &insert = op.Cast<LogicalInsert>();
  if (!dynamic_cast<LanceTableEntry *>(&insert.table) ||
      insert.children.empty()) {
    return;
  }
  auto &projection = *insert.children[0];
  if (projection.type != LogicalOperatorType::LOGICAL_PROJECTION) {
    return;
  }
  auto physical_count = insert.table.GetColumns().PhysicalColumnCount();
  for (idx_t i = 0; i < projection.expressions.size(); i++) {
    if (i >= physical_count ||
        insert.table.GetColumns().GetColumn(PhysicalIndex(i)).Type().id() !=
            LogicalTypeId::BLOB) {
      continue;
    }
    auto &expression = projection.expressions[i];
    if (!expression || !BoundCastExpression::IsCast(*expression)) {
      continue;
    }
    if (expression->GetReturnType().id() != LogicalTypeId::BLOB) {
      continue;
    }
    auto &cast = expression->Cast<BoundFunctionExpression>();
    if (!LanceIsBlobInput(BoundCastExpression::SourceType(cast))) {
      continue;
    }
    expression = std::move(BoundCastExpression::ChildMutable(cast));
  }
}

static void LanceBlobPostBind(PlannerExtensionInput &,
                              BoundStatement &statement) {
  if (statement.plan) {
    NormalizeLanceInsertBlobCasts(*statement.plan);
    LanceWrapMergeReturningOperator(*statement.plan);
  }
}

void RegisterLanceBlobInput(ExtensionLoader &loader) {
  loader.RegisterType(LANCE_BLOB_INPUT_TYPE_NAME, LanceBlobInputType());

  ScalarFunction ref_function("lance_blob_ref", {LogicalType::VARCHAR},
                              LanceBlobInputType(), LanceBlobRefFunction);
  ref_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
  ref_function.SetFallible();
  loader.RegisterFunction(ref_function);
  ScalarFunction ref_range_function(
      "lance_blob_ref",
      {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
      LanceBlobInputType(), LanceBlobRefFunction);
  ref_range_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
  ref_range_function.SetFallible();
  loader.RegisterFunction(ref_range_function);

  ScalarFunction ingest_function("lance_blob_ingest", {LogicalType::VARCHAR},
                                 LanceBlobInputType(), LanceBlobIngestFunction);
  ingest_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
  ingest_function.SetFallible();
  loader.RegisterFunction(ingest_function);
  ScalarFunction ingest_range_function(
      "lance_blob_ingest",
      {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
      LanceBlobInputType(), LanceBlobIngestFunction);
  ingest_range_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
  ingest_range_function.SetFallible();
  loader.RegisterFunction(ingest_range_function);

  // Ordinary values become the bytes variant only. The VARCHAR assignment
  // conversion keeps DuckDB's own string-to-BLOB rules; it never interprets a
  // string as a URI. A negative implicit cost keeps this out of unrelated
  // casts. There is deliberately no input-to-BLOB cast.
  loader.RegisterCastFunction(LogicalType::BLOB, LanceBlobInputType(),
                              BoundCastInfo(LanceBlobBytesCast), 1);
  loader.RegisterCastFunction(LogicalType::VARCHAR, LanceBlobInputType(),
                              BoundCastInfo(LanceBlobBytesCast), -1);
}

void RegisterLanceBlobPlannerExtension(DBConfig &config) {
  PlannerExtension extension;
  extension.post_bind_function = LanceBlobPostBind;
  PlannerExtension::Register(config, std::move(extension));
}

} // namespace duckdb
