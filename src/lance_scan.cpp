#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/table_filter.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>

// FFI ownership contract (Arrow C Data Interface):
// `lance_get_schema` returns an opaque schema handle; caller frees it via
// `lance_free_schema` exactly once.
// `lance_schema_to_arrow` populates `out_schema` on success (return 0) and
// transfers ownership of the ArrowSchema to the caller, who must call
// `out_schema->release(out_schema)` exactly once (or wrap it in RAII).
// `lance_create_stream` and `lance_create_fragment_stream` return opaque stream
// handles; caller closes them via `lance_close_stream` exactly once.
// `lance_stream_next` returns an opaque RecordBatch handle; caller frees it via
// `lance_free_batch` exactly once after use.
// `lance_batch_to_arrow` populates `out_array` and `out_schema` on success
// (return 0) and transfers ownership of both to the caller, who must call
// `release` exactly once on each.
// On error, the callee leaves output `ArrowSchema` / `ArrowArray` untouched; do
// not call `release` unless the caller initialized them to a valid value.
extern "C" {
void *lance_open_dataset(const char *path);
void *lance_open_dataset_with_storage_options(const char *path,
                                              const char **option_keys,
                                              const char **option_values,
                                              size_t options_len);
void lance_close_dataset(void *dataset);

void *lance_get_schema(void *dataset);
void lance_free_schema(void *schema);
int32_t lance_schema_to_arrow(void *schema, ArrowSchema *out_schema);

void *lance_create_stream(void *dataset);
int32_t lance_stream_next(void *stream, void **out_batch);
void lance_close_stream(void *stream);

int32_t lance_last_error_code();
const char *lance_last_error_message();
void lance_free_string(const char *s);

int64_t lance_dataset_count_rows(void *dataset);

uint64_t *lance_dataset_list_fragments(void *dataset, size_t *out_len);
void lance_free_fragment_list(uint64_t *ptr, size_t len);
void *lance_create_fragment_stream(void *dataset, uint64_t fragment_id,
                                   const char **columns, size_t columns_len,
                                   const char *filter_sql);
void *lance_create_fragment_stream_ir(void *dataset, uint64_t fragment_id,
                                      const char **columns, size_t columns_len,
                                      const uint8_t *filter_ir,
                                      size_t filter_ir_len);

const char *lance_explain_dataset_scan_ir(void *dataset, const char **columns,
                                          size_t columns_len,
                                          const uint8_t *filter_ir,
                                          size_t filter_ir_len,
                                          uint8_t verbose);

void lance_free_batch(void *batch);
int64_t lance_batch_num_rows(void *batch);
int32_t lance_batch_to_arrow(void *batch, ArrowArray *out_array,
                             ArrowSchema *out_schema);
}

namespace duckdb {

static string LanceConsumeLastError() {
  auto code = lance_last_error_code();
  string message;
  if (auto *ptr = lance_last_error_message()) {
    message = ptr;
    lance_free_string(ptr);
  }

  if (code == 0 && message.empty()) {
    return "";
  }
  if (message.empty()) {
    return "code=" + to_string(code);
  }
  if (code == 0) {
    return message;
  }
  return message + " (code=" + to_string(code) + ")";
}

static string LanceFormatErrorSuffix() {
  auto err = LanceConsumeLastError();
  if (err.empty()) {
    return "";
  }
  return " (Lance error: " + err + ")";
}

static bool TryLanceExplainDatasetScan(void *dataset,
                                       const vector<string> *columns,
                                       const string *filter_ir, bool verbose,
                                       string &out_plan, string &out_error) {
  out_plan.clear();
  out_error.clear();

  if (!dataset) {
    out_error = "dataset is null";
    return false;
  }

  vector<const char *> col_ptrs;
  if (columns) {
    col_ptrs.reserve(columns->size());
    for (auto &col : *columns) {
      col_ptrs.push_back(col.c_str());
    }
  }

  const uint8_t *filter_ptr = nullptr;
  size_t filter_len = 0;
  if (filter_ir && !filter_ir->empty()) {
    filter_ptr = reinterpret_cast<const uint8_t *>(filter_ir->data()); // NOLINT
    filter_len = filter_ir->size();
  }

  auto *plan_ptr = lance_explain_dataset_scan_ir(
      dataset, col_ptrs.empty() ? nullptr : col_ptrs.data(), col_ptrs.size(),
      filter_ptr, filter_len, verbose ? 1 : 0);
  if (!plan_ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }

  out_plan = plan_ptr;
  lance_free_string(plan_ptr);
  return true;
}

static string NormalizeS3Scheme(const string &path) {
  if (StringUtil::StartsWith(path, "s3a://")) {
    return "s3://" + path.substr(6);
  }
  if (StringUtil::StartsWith(path, "s3n://")) {
    return "s3://" + path.substr(6);
  }
  return path;
}

static string SecretValueToString(const Value &value) {
  if (value.IsNull()) {
    return "";
  }
  return value.ToString();
}

static void AddIfNotEmpty(vector<string> &keys, vector<string> &values,
                          const string &key, const string &value) {
  if (value.empty()) {
    return;
  }
  keys.push_back(key);
  values.push_back(value);
}

static void FillS3StorageOptionsFromSecrets(ClientContext &context,
                                            const string &path,
                                            vector<string> &out_keys,
                                            vector<string> &out_values) {
  auto &secret_manager = SecretManager::Get(context);
  auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
  auto secret_match = secret_manager.LookupSecret(transaction, path, "s3");
  if (!secret_match.HasMatch() || !secret_match.secret_entry ||
      !secret_match.secret_entry->secret) {
    return;
  }

  auto *kv_secret = dynamic_cast<const KeyValueSecret *>(
      secret_match.secret_entry->secret.get());
  if (!kv_secret) {
    return;
  }

  auto key_id = SecretValueToString(kv_secret->TryGetValue("key_id"));
  auto secret_access_key =
      SecretValueToString(kv_secret->TryGetValue("secret"));
  auto session_token =
      SecretValueToString(kv_secret->TryGetValue("session_token"));
  auto region = SecretValueToString(kv_secret->TryGetValue("region"));
  auto endpoint = SecretValueToString(kv_secret->TryGetValue("endpoint"));
  auto url_style = SecretValueToString(kv_secret->TryGetValue("url_style"));
  auto use_ssl = SecretValueToString(kv_secret->TryGetValue("use_ssl"));

  if (key_id.empty() && secret_access_key.empty()) {
    AddIfNotEmpty(out_keys, out_values, "skip_signature", "true");
  } else {
    AddIfNotEmpty(out_keys, out_values, "access_key_id", key_id);
    AddIfNotEmpty(out_keys, out_values, "secret_access_key", secret_access_key);
    AddIfNotEmpty(out_keys, out_values, "session_token", session_token);
  }

  AddIfNotEmpty(out_keys, out_values, "region", region);
  AddIfNotEmpty(out_keys, out_values, "endpoint", endpoint);

  if (StringUtil::CIEquals(url_style, "vhost") ||
      StringUtil::CIEquals(url_style, "virtual_hosted")) {
    AddIfNotEmpty(out_keys, out_values, "virtual_hosted_style_request", "true");
  } else if (StringUtil::CIEquals(url_style, "path")) {
    AddIfNotEmpty(out_keys, out_values, "virtual_hosted_style_request",
                  "false");
  }

  if (!use_ssl.empty()) {
    if (StringUtil::CIEquals(use_ssl, "false") ||
        StringUtil::CIEquals(use_ssl, "0")) {
      AddIfNotEmpty(out_keys, out_values, "allow_http", "true");
    }
  }
}

struct LanceScanBindData : public TableFunctionData {
  string file_path;
  bool explain_verbose = false;
  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;
  vector<string> lance_pushed_filter_ir_parts;

  ~LanceScanBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
    }
  }
};

struct LanceScanGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> next_fragment_idx{0};
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  std::atomic<idx_t> streams_opened{0};
  std::atomic<idx_t> filter_pushdown_fallbacks{0};

  vector<uint64_t> fragment_ids;
  idx_t max_threads = 1;

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;

  vector<string> scan_column_names;
  string lance_filter_ir;

  bool count_only = false;
  idx_t count_only_total_rows = 0;
  std::atomic<idx_t> count_only_offset{0};

  std::atomic<bool> explain_computed{false};
  string explain_plan;
  string explain_error;
  std::mutex explain_mutex;

  idx_t MaxThreads() const override { return max_threads; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceScanLocalState : public ArrowScanLocalState {
  explicit LanceScanLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                               ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  LanceScanGlobalState *global_state = nullptr;
  idx_t fragment_pos = 0;
  SelectionVector filter_sel;

  ~LanceScanLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};
static bool LanceSupportsPushdownLogicalType(const LogicalType &type) {
  switch (type.id()) {
  case LogicalTypeId::BOOLEAN:
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::BIGINT:
  case LogicalTypeId::DATE:
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::UBIGINT:
  case LogicalTypeId::FLOAT:
  case LogicalTypeId::DOUBLE:
  case LogicalTypeId::VARCHAR:
    return true;
  default:
    return false;
  }
}

static bool LanceSupportsPushdownType(const FunctionData &bind_data,
                                      idx_t col_idx) {
  auto &scan_bind = bind_data.Cast<LanceScanBindData>();
  if (col_idx >= scan_bind.types.size()) {
    return false;
  }
  return LanceSupportsPushdownLogicalType(scan_bind.types[col_idx]);
}

static constexpr char LANCE_FILTER_IR_MAGIC[] = {'L', 'F', 'T', '1'};
static constexpr uint8_t LANCE_FILTER_IR_VERSION = 1;

enum class LanceFilterIRTag : uint8_t {
  COLUMN_REF = 1,
  LITERAL = 2,
  AND = 3,
  OR = 4,
  NOT = 5,
  COMPARISON = 6,
  IS_NULL = 7,
  IS_NOT_NULL = 8,
  IN_LIST = 9,
};

enum class LanceFilterIRLiteralTag : uint8_t {
  NULL_VALUE = 0,
  BOOL = 1,
  I64 = 2,
  U64 = 3,
  F32 = 4,
  F64 = 5,
  STRING = 6,
};

enum class LanceFilterIRComparisonOp : uint8_t {
  EQ = 0,
  NOT_EQ = 1,
  LT = 2,
  LT_EQ = 3,
  GT = 4,
  GT_EQ = 5,
};

static void LanceFilterIRAppendU8(string &out, uint8_t v) {
  out.push_back(static_cast<char>(v));
}

static void LanceFilterIRAppendU32(string &out, uint32_t v) {
  uint8_t buf[4];
  buf[0] = static_cast<uint8_t>(v & 0xFF);
  buf[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  buf[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  buf[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
  out.append(reinterpret_cast<const char *>(buf), sizeof(buf));
}

static void LanceFilterIRAppendU64(string &out, uint64_t v) {
  uint8_t buf[8];
  for (idx_t i = 0; i < 8; i++) {
    buf[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
  }
  out.append(reinterpret_cast<const char *>(buf), sizeof(buf));
}

static void LanceFilterIRAppendI64(string &out, int64_t v) {
  LanceFilterIRAppendU64(out, static_cast<uint64_t>(v));
}

static void LanceFilterIRAppendF32(string &out, float v) {
  uint32_t bits = 0;
  memcpy(&bits, &v, sizeof(bits));
  LanceFilterIRAppendU32(out, bits);
}

static void LanceFilterIRAppendF64(string &out, double v) {
  uint64_t bits = 0;
  memcpy(&bits, &v, sizeof(bits));
  LanceFilterIRAppendU64(out, bits);
}

static bool LanceFilterIRAppendLenPrefixed(string &out, const string &bytes) {
  if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  LanceFilterIRAppendU32(out, static_cast<uint32_t>(bytes.size()));
  out.append(bytes);
  return true;
}

static bool LanceFilterIRAppendLenPrefixedString(string &out,
                                                 const string &value) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  LanceFilterIRAppendU32(out, static_cast<uint32_t>(value.size()));
  out.append(value);
  return true;
}

static string LanceFilterIREncodeMessage(const string &root_node) {
  string out;
  out.append(LANCE_FILTER_IR_MAGIC, sizeof(LANCE_FILTER_IR_MAGIC));
  LanceFilterIRAppendU8(out, LANCE_FILTER_IR_VERSION);
  out.append(root_node);
  return out;
}

static bool SplitLanceColumnPath(const string &name, vector<string> &segments) {
  segments.clear();
  idx_t start = 0;
  for (idx_t i = 0; i <= name.size(); i++) {
    if (i == name.size() || name[i] == '.') {
      if (i == start) {
        return false;
      }
      segments.push_back(name.substr(start, i - start));
      start = i + 1;
    }
  }
  return !segments.empty();
}

static bool TryEncodeLanceFilterIRColumnRef(const string &name,
                                            string &out_ir) {
  vector<string> segments;
  if (!SplitLanceColumnPath(name, segments)) {
    return false;
  }

  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::COLUMN_REF));
  if (segments.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  LanceFilterIRAppendU32(out_ir, static_cast<uint32_t>(segments.size()));
  for (auto &segment : segments) {
    if (!LanceFilterIRAppendLenPrefixedString(out_ir, segment)) {
      return false;
    }
  }
  return true;
}

static bool TryEncodeLanceFilterIRLiteral(const Value &value, string &out_ir) {
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::LITERAL));

  if (value.IsNull()) {
    LanceFilterIRAppendU8(
        out_ir, static_cast<uint8_t>(LanceFilterIRLiteralTag::NULL_VALUE));
    return true;
  }

  switch (value.type().id()) {
  case LogicalTypeId::BOOLEAN:
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::BOOL));
    LanceFilterIRAppendU8(out_ir, value.GetValue<bool>() ? 1 : 0);
    return true;
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::BIGINT:
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::I64));
    LanceFilterIRAppendI64(out_ir, value.GetValue<int64_t>());
    return true;
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::UBIGINT:
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::U64));
    LanceFilterIRAppendU64(out_ir, value.GetValue<uint64_t>());
    return true;
  case LogicalTypeId::FLOAT: {
    auto v = value.GetValue<float>();
    if (!std::isfinite(v)) {
      return false;
    }
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::F32));
    LanceFilterIRAppendF32(out_ir, v);
    return true;
  }
  case LogicalTypeId::DOUBLE: {
    auto v = value.GetValue<double>();
    if (!std::isfinite(v)) {
      return false;
    }
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::F64));
    LanceFilterIRAppendF64(out_ir, v);
    return true;
  }
  case LogicalTypeId::VARCHAR: {
    auto v = value.GetValue<string>();
    LanceFilterIRAppendU8(
        out_ir, static_cast<uint8_t>(LanceFilterIRLiteralTag::STRING));
    return LanceFilterIRAppendLenPrefixedString(out_ir, v);
  }
  default:
    return false;
  }
}

static bool TryEncodeLanceFilterIRComparisonOp(ExpressionType type,
                                               uint8_t &out_op) {
  switch (type) {
  case ExpressionType::COMPARE_EQUAL:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::EQ);
    return true;
  case ExpressionType::COMPARE_NOTEQUAL:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::NOT_EQ);
    return true;
  case ExpressionType::COMPARE_LESSTHAN:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::LT);
    return true;
  case ExpressionType::COMPARE_LESSTHANOREQUALTO:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::LT_EQ);
    return true;
  case ExpressionType::COMPARE_GREATERTHAN:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::GT);
    return true;
  case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::GT_EQ);
    return true;
  default:
    return false;
  }
}

static bool TryEncodeLanceFilterIRConjunction(LanceFilterIRTag tag,
                                              const vector<string> &children,
                                              string &out_ir) {
  if (children.empty()) {
    return false;
  }
  if (children.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir, static_cast<uint8_t>(tag));
  LanceFilterIRAppendU32(out_ir, static_cast<uint32_t>(children.size()));
  for (auto &child : children) {
    if (!LanceFilterIRAppendLenPrefixed(out_ir, child)) {
      return false;
    }
  }
  return true;
}

static bool TryEncodeLanceFilterIRUnary(LanceFilterIRTag tag,
                                        const string &child, string &out_ir) {
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir, static_cast<uint8_t>(tag));
  return LanceFilterIRAppendLenPrefixed(out_ir, child);
}

static bool TryEncodeLanceFilterIRComparison(uint8_t op, const string &left,
                                             const string &right,
                                             string &out_ir) {
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::COMPARISON));
  LanceFilterIRAppendU8(out_ir, op);
  return LanceFilterIRAppendLenPrefixed(out_ir, left) &&
         LanceFilterIRAppendLenPrefixed(out_ir, right);
}

static bool TryEncodeLanceFilterIRInList(bool negated, const string &expr,
                                         const vector<string> &list,
                                         string &out_ir) {
  if (list.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::IN_LIST));
  LanceFilterIRAppendU8(out_ir, negated ? 1 : 0);
  if (!LanceFilterIRAppendLenPrefixed(out_ir, expr)) {
    return false;
  }
  LanceFilterIRAppendU32(out_ir, static_cast<uint32_t>(list.size()));
  for (auto &item : list) {
    if (!LanceFilterIRAppendLenPrefixed(out_ir, item)) {
      return false;
    }
  }
  return true;
}

static bool TryBuildLanceTableFilterIRExpr(const string &col_ref_ir,
                                           const TableFilter &filter,
                                           string &out_ir) {
  switch (filter.filter_type) {
  case TableFilterType::CONSTANT_COMPARISON: {
    auto &f = filter.Cast<ConstantFilter>();
    uint8_t op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(f.comparison_type, op)) {
      return false;
    }
    string lit_ir;
    if (!TryEncodeLanceFilterIRLiteral(f.constant, lit_ir)) {
      return false;
    }
    return TryEncodeLanceFilterIRComparison(op, col_ref_ir, lit_ir, out_ir);
  }
  case TableFilterType::IS_NULL:
    return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::IS_NULL, col_ref_ir,
                                       out_ir);
  case TableFilterType::IS_NOT_NULL:
    return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::IS_NOT_NULL,
                                       col_ref_ir, out_ir);
  case TableFilterType::IN_FILTER: {
    auto &f = filter.Cast<InFilter>();
    vector<string> items;
    items.reserve(f.values.size());
    for (auto &value : f.values) {
      string lit_ir;
      if (!TryEncodeLanceFilterIRLiteral(value, lit_ir)) {
        return false;
      }
      items.push_back(std::move(lit_ir));
    }
    return TryEncodeLanceFilterIRInList(false, col_ref_ir, items, out_ir);
  }
  case TableFilterType::CONJUNCTION_AND: {
    auto &f = filter.Cast<ConjunctionAndFilter>();
    vector<string> children;
    children.reserve(f.child_filters.size());
    for (auto &child : f.child_filters) {
      string child_ir;
      if (!TryBuildLanceTableFilterIRExpr(col_ref_ir, *child, child_ir)) {
        return false;
      }
      children.push_back(std::move(child_ir));
    }
    return TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND, children,
                                             out_ir);
  }
  case TableFilterType::CONJUNCTION_OR: {
    auto &f = filter.Cast<ConjunctionOrFilter>();
    vector<string> children;
    children.reserve(f.child_filters.size());
    for (auto &child : f.child_filters) {
      string child_ir;
      if (!TryBuildLanceTableFilterIRExpr(col_ref_ir, *child, child_ir)) {
        return false;
      }
      children.push_back(std::move(child_ir));
    }
    return TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::OR, children,
                                             out_ir);
  }
  default:
    return false;
  }
}

static bool BuildLanceTableFilterIRParts(const LanceScanBindData &bind_data,
                                         const TableFunctionInitInput &input,
                                         vector<string> &out_parts) {
  if (!input.filters || input.filters->filters.empty()) {
    return true;
  }
  out_parts.clear();
  out_parts.reserve(input.filters->filters.size());

  for (auto &it : input.filters->filters) {
    auto scan_col_idx = it.first;
    auto &filter = *it.second;
    if (scan_col_idx >= input.column_ids.size()) {
      return false;
    }
    auto col_id = input.column_ids[scan_col_idx];
    if (col_id >= bind_data.names.size()) {
      return false;
    }
    string col_ref_ir;
    if (!TryEncodeLanceFilterIRColumnRef(bind_data.names[col_id], col_ref_ir)) {
      return false;
    }
    string filter_ir;
    if (!TryBuildLanceTableFilterIRExpr(col_ref_ir, filter, filter_ir)) {
      return false;
    }
    out_parts.push_back(std::move(filter_ir));
  }

  return true;
}

static bool TryBuildLanceExprColumnRefIR(const LogicalGet &get,
                                         const LanceScanBindData &bind_data,
                                         const Expression &expr,
                                         string &out_ir) {
  if (expr.expression_class != ExpressionClass::BOUND_COLUMN_REF) {
    return false;
  }
  auto &colref = expr.Cast<BoundColumnRefExpression>();
  if (colref.depth != 0) {
    return false;
  }
  if (colref.binding.table_index != get.table_index) {
    return false;
  }
  auto &column_ids = get.GetColumnIds();
  if (colref.binding.column_index >= column_ids.size()) {
    return false;
  }
  auto &col_index = column_ids[colref.binding.column_index];
  if (col_index.IsVirtualColumn()) {
    return false;
  }
  if (!col_index.GetChildIndexes().empty()) {
    return false;
  }
  auto col_id = col_index.GetPrimaryIndex();
  if (col_id >= bind_data.names.size() || col_id >= bind_data.types.size()) {
    return false;
  }
  if (!LanceSupportsPushdownLogicalType(bind_data.types[col_id])) {
    return false;
  }
  return TryEncodeLanceFilterIRColumnRef(bind_data.names[col_id], out_ir);
}

static bool TryBuildLanceExprFilterIR(const LogicalGet &get,
                                      const LanceScanBindData &bind_data,
                                      const Expression &expr, string &out_ir) {
  switch (expr.expression_class) {
  case ExpressionClass::BOUND_COLUMN_REF:
    return TryBuildLanceExprColumnRefIR(get, bind_data, expr, out_ir);
  case ExpressionClass::BOUND_CONSTANT: {
    auto &c = expr.Cast<BoundConstantExpression>();
    return TryEncodeLanceFilterIRLiteral(c.value, out_ir);
  }
  case ExpressionClass::BOUND_CAST: {
    auto &cast = expr.Cast<BoundCastExpression>();
    if (cast.try_cast) {
      return false;
    }
    if (cast.child->expression_class != ExpressionClass::BOUND_CONSTANT) {
      return false;
    }
    auto &c = cast.child->Cast<BoundConstantExpression>();
    return TryEncodeLanceFilterIRLiteral(c.value, out_ir);
  }
  case ExpressionClass::BOUND_COMPARISON: {
    auto &cmp = expr.Cast<BoundComparisonExpression>();
    if (cmp.type == ExpressionType::COMPARE_DISTINCT_FROM ||
        cmp.type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      return false;
    }
    uint8_t op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(cmp.type, op)) {
      return false;
    }
    string lhs_ir, rhs_ir;
    if (!TryBuildLanceExprFilterIR(get, bind_data, *cmp.left, lhs_ir) ||
        !TryBuildLanceExprFilterIR(get, bind_data, *cmp.right, rhs_ir)) {
      return false;
    }
    return TryEncodeLanceFilterIRComparison(op, lhs_ir, rhs_ir, out_ir);
  }
  case ExpressionClass::BOUND_CONJUNCTION: {
    auto &conj = expr.Cast<BoundConjunctionExpression>();
    LanceFilterIRTag tag;
    if (conj.type == ExpressionType::CONJUNCTION_AND) {
      tag = LanceFilterIRTag::AND;
    } else if (conj.type == ExpressionType::CONJUNCTION_OR) {
      tag = LanceFilterIRTag::OR;
    } else {
      return false;
    }
    vector<string> children;
    children.reserve(conj.children.size());
    for (auto &child : conj.children) {
      string child_ir;
      if (!TryBuildLanceExprFilterIR(get, bind_data, *child, child_ir)) {
        return false;
      }
      children.push_back(std::move(child_ir));
    }
    return TryEncodeLanceFilterIRConjunction(tag, children, out_ir);
  }
  case ExpressionClass::BOUND_OPERATOR: {
    auto &op = expr.Cast<BoundOperatorExpression>();
    if (op.type == ExpressionType::OPERATOR_NOT) {
      if (op.children.size() != 1) {
        return false;
      }
      string child_ir;
      if (!TryBuildLanceExprFilterIR(get, bind_data, *op.children[0],
                                     child_ir)) {
        return false;
      }
      return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::NOT, child_ir,
                                         out_ir);
    }
    if (op.type == ExpressionType::OPERATOR_IS_NULL ||
        op.type == ExpressionType::OPERATOR_IS_NOT_NULL) {
      if (op.children.size() != 1) {
        return false;
      }
      string child_ir;
      if (!TryBuildLanceExprFilterIR(get, bind_data, *op.children[0],
                                     child_ir)) {
        return false;
      }
      return TryEncodeLanceFilterIRUnary(
          op.type == ExpressionType::OPERATOR_IS_NULL
              ? LanceFilterIRTag::IS_NULL
              : LanceFilterIRTag::IS_NOT_NULL,
          child_ir, out_ir);
    }
    if (op.type == ExpressionType::COMPARE_IN ||
        op.type == ExpressionType::COMPARE_NOT_IN) {
      if (op.children.size() < 2) {
        return false;
      }
      string lhs_ir;
      if (!TryBuildLanceExprFilterIR(get, bind_data, *op.children[0], lhs_ir)) {
        return false;
      }
      vector<string> values;
      values.reserve(op.children.size() - 1);
      for (idx_t i = 1; i < op.children.size(); i++) {
        if (op.children[i]->expression_class !=
            ExpressionClass::BOUND_CONSTANT) {
          return false;
        }
        auto &c = op.children[i]->Cast<BoundConstantExpression>();
        string lit_ir;
        if (!TryEncodeLanceFilterIRLiteral(c.value, lit_ir)) {
          return false;
        }
        values.push_back(std::move(lit_ir));
      }
      return TryEncodeLanceFilterIRInList(
          op.type == ExpressionType::COMPARE_NOT_IN, lhs_ir, values, out_ir);
    }
    return false;
  }
  case ExpressionClass::BOUND_BETWEEN: {
    auto &between = expr.Cast<BoundBetweenExpression>();
    uint8_t lower_op = 0;
    uint8_t upper_op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(between.LowerComparisonType(),
                                            lower_op) ||
        !TryEncodeLanceFilterIRComparisonOp(between.UpperComparisonType(),
                                            upper_op)) {
      return false;
    }
    string input_ir, lower_ir, upper_ir;
    if (!TryBuildLanceExprFilterIR(get, bind_data, *between.input, input_ir) ||
        !TryBuildLanceExprFilterIR(get, bind_data, *between.lower, lower_ir) ||
        !TryBuildLanceExprFilterIR(get, bind_data, *between.upper, upper_ir)) {
      return false;
    }
    string lower_cmp_ir;
    if (!TryEncodeLanceFilterIRComparison(lower_op, input_ir, lower_ir,
                                          lower_cmp_ir)) {
      return false;
    }
    string upper_cmp_ir;
    if (!TryEncodeLanceFilterIRComparison(upper_op, input_ir, upper_ir,
                                          upper_cmp_ir)) {
      return false;
    }
    vector<string> children;
    children.push_back(std::move(lower_cmp_ir));
    children.push_back(std::move(upper_cmp_ir));
    return TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND, children,
                                             out_ir);
  }
  default:
    return false;
  }
}

static void
LancePushdownComplexFilter(ClientContext &context, LogicalGet &get,
                           FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
  if (!bind_data || filters.empty()) {
    return;
  }
  auto &scan_bind = bind_data->Cast<LanceScanBindData>();

  for (auto &expr : filters) {
    if (!expr || expr->HasParameter() || expr->IsVolatile() ||
        expr->CanThrow()) {
      continue;
    }
    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind, *expr, filter_ir)) {
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));
  }
}

static unique_ptr<FunctionData> LanceScanBind(ClientContext &context,
                                              TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types,
                                              vector<string> &names) {
  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw InvalidInputException("lance_scan requires a dataset root path");
  }

  auto result = make_uniq<LanceScanBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  auto verbose_it = input.named_parameters.find("explain_verbose");
  if (verbose_it != input.named_parameters.end() &&
      !verbose_it->second.IsNull()) {
    result->explain_verbose =
        verbose_it->second.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  auto open_path = result->file_path;
  vector<string> option_keys;
  vector<string> option_values;

  if (StringUtil::StartsWith(open_path, "s3://") ||
      StringUtil::StartsWith(open_path, "s3a://") ||
      StringUtil::StartsWith(open_path, "s3n://")) {
    open_path = NormalizeS3Scheme(open_path);
    FillS3StorageOptionsFromSecrets(context, open_path, option_keys,
                                    option_values);
  }

  if (!option_keys.empty()) {
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    key_ptrs.reserve(option_keys.size());
    value_ptrs.reserve(option_values.size());
    for (idx_t i = 0; i < option_keys.size(); i++) {
      key_ptrs.push_back(option_keys[i].c_str());
      value_ptrs.push_back(option_values[i].c_str());
    }
    result->dataset = lance_open_dataset_with_storage_options(
        open_path.c_str(), key_ptrs.data(), value_ptrs.data(),
        option_keys.size());
  } else {
    result->dataset = lance_open_dataset(open_path.c_str());
  }
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_schema(result->dataset);
  if (!schema_handle) {
    throw IOException("Failed to get schema from Lance dataset: " +
                      result->file_path + LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);

  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
  auto state = make_uniq_base<GlobalTableFunctionState, LanceScanGlobalState>();
  auto &scan_state = state->Cast<LanceScanGlobalState>();

  scan_state.projection_ids = input.projection_ids;
  if (!input.projection_ids.empty()) {
    scan_state.scanned_types.reserve(input.column_ids.size());
    for (auto col_id : input.column_ids) {
      if (col_id >= bind_data.types.size()) {
        throw IOException("Invalid column id in projection");
      }
      scan_state.scanned_types.push_back(bind_data.types[col_id]);
    }
  }

  scan_state.scan_column_names.reserve(input.column_ids.size());
  for (auto col_id : input.column_ids) {
    if (col_id >= bind_data.names.size()) {
      throw IOException("Invalid column id in projection");
    }
    scan_state.scan_column_names.push_back(bind_data.names[col_id]);
  }

  vector<string> filter_parts;

  vector<string> table_filter_parts;
  auto table_filters_ok =
      BuildLanceTableFilterIRParts(bind_data, input, table_filter_parts);
  if (table_filters_ok) {
    filter_parts = std::move(table_filter_parts);
  }

  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    filter_parts.reserve(filter_parts.size() +
                         bind_data.lance_pushed_filter_ir_parts.size());
    for (auto &part : bind_data.lance_pushed_filter_ir_parts) {
      filter_parts.push_back(part);
    }
  }

  if (!filter_parts.empty()) {
    string root_node;
    if (filter_parts.size() == 1) {
      root_node = std::move(filter_parts[0]);
    } else if (!TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND,
                                                  filter_parts, root_node)) {
      root_node.clear();
    }
    if (!root_node.empty()) {
      scan_state.lance_filter_ir = LanceFilterIREncodeMessage(root_node);
    }
  }

  if (scan_state.scan_column_names.empty() &&
      scan_state.lance_filter_ir.empty()) {
    auto rows = lance_dataset_count_rows(bind_data.dataset);
    if (rows < 0) {
      throw IOException("Failed to count Lance rows" +
                        LanceFormatErrorSuffix());
    }
    scan_state.count_only = true;
    scan_state.count_only_total_rows = NumericCast<idx_t>(rows);
    scan_state.max_threads = 1;
    return state;
  }

  size_t fragment_count = 0;
  auto fragments_ptr =
      lance_dataset_list_fragments(bind_data.dataset, &fragment_count);
  if (!fragments_ptr) {
    throw IOException("Failed to list Lance fragments" +
                      LanceFormatErrorSuffix());
  }
  scan_state.fragment_ids.assign(fragments_ptr, fragments_ptr + fragment_count);
  lance_free_fragment_list(fragments_ptr, fragment_count);

  auto threads = context.db->NumberOfThreads();
  scan_state.max_threads = MaxValue<idx_t>(
      1, MinValue<idx_t>(threads, scan_state.fragment_ids.size()));

  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceScanLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                   GlobalTableFunctionState *global_state) {
  auto &scan_global = global_state->Cast<LanceScanGlobalState>();
  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result =
      make_uniq<LanceScanLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &scan_global;
  if (scan_global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, scan_global.scanned_types);
  }
  if (scan_global.count_only) {
    return std::move(result);
  }
  // Early stop: no fragments left for this thread.
  auto fragment_pos = scan_global.next_fragment_idx.fetch_add(1);
  if (fragment_pos >= scan_global.fragment_ids.size()) {
    return nullptr;
  }
  result->fragment_pos = fragment_pos;
  return std::move(result);
}

static bool LanceScanOpenStream(ClientContext &context,
                                const LanceScanBindData &bind_data,
                                LanceScanGlobalState &global_state,
                                LanceScanLocalState &local_state) {
  if (local_state.stream) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
  }

  if (local_state.fragment_pos >= global_state.fragment_ids.size()) {
    return false;
  }
  auto fragment_id = global_state.fragment_ids[local_state.fragment_pos];

  vector<const char *> columns;
  columns.reserve(global_state.scan_column_names.size());
  for (auto &name : global_state.scan_column_names) {
    columns.push_back(name.c_str());
  }

  const uint8_t *filter_ir = global_state.lance_filter_ir.empty()
                                 ? nullptr
                                 : reinterpret_cast<const uint8_t *>(
                                       global_state.lance_filter_ir.data());
  auto filter_ir_len = global_state.lance_filter_ir.size();

  auto stream = lance_create_fragment_stream_ir(bind_data.dataset, fragment_id,
                                                columns.data(), columns.size(),
                                                filter_ir, filter_ir_len);
  if (!stream && filter_ir) {
    // Best-effort: if filter pushdown failed, retry without it and rely on
    // DuckDB-side filter execution for correctness.
    global_state.filter_pushdown_fallbacks.fetch_add(1);
    stream = lance_create_fragment_stream_ir(bind_data.dataset, fragment_id,
                                             columns.data(), columns.size(),
                                             nullptr, 0);
  }
  if (!stream) {
    throw IOException("Failed to create Lance fragment stream" +
                      LanceFormatErrorSuffix());
  }
  global_state.streams_opened.fetch_add(1);
  local_state.stream = stream;
  return true;
}

static bool LanceScanLoadNextBatch(LanceScanLocalState &local_state) {
  if (!local_state.stream) {
    return false;
  }
  void *batch = nullptr;
  auto rc = lance_stream_next(local_state.stream, &batch);
  if (rc == 1) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
    return false;
  }
  if (rc != 0) {
    throw IOException("Failed to read next Lance RecordBatch" +
                      LanceFormatErrorSuffix());
  }

  auto new_chunk = make_shared_ptr<ArrowArrayWrapper>();
  memset(&new_chunk->arrow_array, 0, sizeof(new_chunk->arrow_array));
  ArrowSchema tmp_schema;
  memset(&tmp_schema, 0, sizeof(tmp_schema));

  if (lance_batch_to_arrow(batch, &new_chunk->arrow_array, &tmp_schema) != 0) {
    lance_free_batch(batch);
    throw IOException(
        "Failed to export Lance RecordBatch to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }

  lance_free_batch(batch);

  if (local_state.global_state) {
    local_state.global_state->record_batches.fetch_add(1);
    auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
    local_state.global_state->record_batch_rows.fetch_add(rows);
  }

  if (tmp_schema.release) {
    tmp_schema.release(&tmp_schema);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                               DataChunk &chunk, SelectionVector &sel) {
  if (chunk.size() == 0) {
    return;
  }
  unique_ptr<Expression> combined;
  for (auto &it : filters.filters) {
    auto scan_col_idx = it.first;
    if (scan_col_idx >= chunk.ColumnCount()) {
      continue;
    }
    BoundReferenceExpression col_expr(chunk.data[scan_col_idx].GetType(),
                                      NumericCast<storage_t>(scan_col_idx));
    auto expr = it.second->ToExpression(col_expr);
    if (!combined) {
      combined = std::move(expr);
    } else {
      auto conj = make_uniq<BoundConjunctionExpression>(
          ExpressionType::CONJUNCTION_AND);
      conj->children.push_back(std::move(combined));
      conj->children.push_back(std::move(expr));
      combined = std::move(conj);
    }
  }
  if (!combined) {
    return;
  }
  ExpressionExecutor executor(context, *combined);
  auto selected = executor.SelectExpression(chunk, sel);
  if (selected != chunk.size()) {
    chunk.Slice(sel, selected);
  }
}

static void LanceScanFunc(ClientContext &context, TableFunctionInput &data,
                          DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceScanBindData>();
  auto &global_state = data.global_state->Cast<LanceScanGlobalState>();
  auto &local_state = data.local_state->Cast<LanceScanLocalState>();

  if (global_state.count_only) {
    auto start = global_state.count_only_offset.fetch_add(STANDARD_VECTOR_SIZE);
    if (start >= global_state.count_only_total_rows) {
      return;
    }
    auto output_size = MinValue<idx_t>(
        STANDARD_VECTOR_SIZE, global_state.count_only_total_rows - start);
    output.SetCardinality(output_size);
    output.Verify();
    return;
  }

  while (true) {
    if (!local_state.stream) {
      if (!LanceScanOpenStream(context, bind_data, global_state, local_state)) {
        return;
      }
    }

    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceScanLoadNextBatch(local_state)) {
        // Stream finished, try next fragment.
        local_state.fragment_pos = global_state.next_fragment_idx.fetch_add(1);
        continue;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    auto start = global_state.lines_read.fetch_add(output_size);

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      local_state.all_columns.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        local_state.all_columns, start);
      local_state.chunk_offset += output_size;
      if (local_state.filters) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(
          local_state, bind_data.arrow_table.GetColumns(), output, start);
      local_state.chunk_offset += output_size;
      if (local_state.filters) {
        ApplyDuckDBFilters(context, *local_state.filters, output,
                           local_state.filter_sel);
      }
    }

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

static InsertionOrderPreservingMap<string>
LanceScanToString(TableFunctionToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";
  result["Lance Pushed Filter Parts"] =
      to_string(bind_data.lance_pushed_filter_ir_parts.size());

  string filter_ir_msg;
  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    vector<string> parts;
    parts.reserve(bind_data.lance_pushed_filter_ir_parts.size());
    for (auto &part : bind_data.lance_pushed_filter_ir_parts) {
      parts.push_back(part);
    }

    string root_node;
    if (parts.size() == 1) {
      root_node = std::move(parts[0]);
    } else if (!TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND, parts,
                                                  root_node)) {
      root_node.clear();
    }

    if (!root_node.empty()) {
      filter_ir_msg = LanceFilterIREncodeMessage(root_node);
    }
  }

  result["Lance Filter IR Bytes (Bind)"] = to_string(filter_ir_msg.size());

  string plan;
  string error;
  if (TryLanceExplainDatasetScan(bind_data.dataset, nullptr,
                                 filter_ir_msg.empty() ? nullptr
                                                       : &filter_ir_msg,
                                 bind_data.explain_verbose, plan, error)) {
    result["Lance Plan (Bind)"] = plan;
  } else if (!error.empty()) {
    result["Lance Plan Error (Bind)"] = error;
  }

  return result;
}

static InsertionOrderPreservingMap<string>
LanceScanDynamicToString(TableFunctionDynamicToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
  auto &global_state = input.global_state->Cast<LanceScanGlobalState>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";
  result["Lance Fragments"] = to_string(global_state.fragment_ids.size());
  result["Lance Max Threads"] = to_string(global_state.max_threads);
  result["Lance Streams Opened"] =
      to_string(global_state.streams_opened.load());
  result["Lance Filter Pushdown Fallbacks"] =
      to_string(global_state.filter_pushdown_fallbacks.load());
  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());

  if (global_state.count_only) {
    result["Lance Count Only"] = "true";
    result["Lance Count Total Rows"] =
        to_string(global_state.count_only_total_rows);
    return result;
  }

  result["Lance Filter IR Bytes"] =
      to_string(global_state.lance_filter_ir.size());
  if (!global_state.scan_column_names.empty()) {
    result["Lance Projection"] =
        StringUtil::Join(global_state.scan_column_names, "\n");
  }

  if (!global_state.explain_computed.load()) {
    std::lock_guard<std::mutex> guard(global_state.explain_mutex);
    if (!global_state.explain_computed.load()) {
      string plan;
      string error;
      auto ok = TryLanceExplainDatasetScan(
          bind_data.dataset, &global_state.scan_column_names,
          global_state.lance_filter_ir.empty() ? nullptr
                                               : &global_state.lance_filter_ir,
          bind_data.explain_verbose, plan, error);
      if (ok) {
        global_state.explain_plan = std::move(plan);
      } else {
        global_state.explain_error = std::move(error);
      }
      global_state.explain_computed.store(true);
    }
  }

  if (!global_state.explain_plan.empty()) {
    result["Lance Plan"] = global_state.explain_plan;
  } else if (!global_state.explain_error.empty()) {
    result["Lance Plan Error"] = global_state.explain_error;
  }

  return result;
}

void RegisterLanceScan(ExtensionLoader &loader) {
  TableFunction lance_scan("lance_scan", {LogicalType::VARCHAR}, LanceScanFunc,
                           LanceScanBind, LanceScanInitGlobal,
                           LanceScanLocalInit);
  lance_scan.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  lance_scan.projection_pushdown = true;
  lance_scan.filter_pushdown = true;
  lance_scan.filter_prune = true;
  lance_scan.supports_pushdown_type = LanceSupportsPushdownType;
  lance_scan.pushdown_complex_filter = LancePushdownComplexFilter;
  lance_scan.to_string = LanceScanToString;
  lance_scan.dynamic_to_string = LanceScanDynamicToString;
  loader.RegisterFunction(lance_scan);
}

} // namespace duckdb
