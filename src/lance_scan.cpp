#include "duckdb/function/partition_stats.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/main/profiler/profiling_node.hpp"
#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/sample_options.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/table_filter.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_exec_ir.hpp"
#include "lance_ffi.hpp"
#include "lance_filter_ir.hpp"
#include "lance_arrow_compat.hpp"
#include "lance_logical_exec.hpp"
#include "lance_scan_bind_data.hpp"
#include "lance_table_entry.hpp"

#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

// FFI ownership contract (Arrow C Data Interface):
// `lance_get_schema` returns an opaque schema handle; caller frees it via
// `lance_free_schema` exactly once.
// `lance_schema_to_arrow` populates `out_schema` on success (return 0) and
// transfers ownership of the ArrowSchema to the caller, who must call
// `out_schema->release(out_schema)` exactly once (or wrap it in RAII).
// `lance_create_fragment_stream_ir` returns an opaque stream handle; caller
// closes it via `lance_close_stream` exactly once.
// `lance_stream_next` returns an opaque RecordBatch handle; caller frees it via
// `lance_free_batch` exactly once after use.
// `lance_batch_to_arrow` populates `out_array` and `out_schema` on success
// (return 0) and transfers ownership of both to the caller, who must call
// `release` exactly once on each.
// On error, the callee leaves output `ArrowSchema` / `ArrowArray` untouched; do
// not call `release` unless the caller initialized them to a valid value.
namespace duckdb {

static TableFunction LanceExecFunction();

static bool LanceFieldPathSegmentNeedsQuoting(const string &segment) {
  if (segment.empty()) {
    return true;
  }
  for (auto c : segment) {
    if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_') {
      return true;
    }
  }
  return false;
}

static string FormatLanceFieldPath(const vector<string> &segments) {
  if (segments.empty()) {
    return "";
  }

  string out;
  for (idx_t i = 0; i < segments.size(); i++) {
    if (i > 0) {
      out.push_back('.');
    }

    auto &segment = segments[i];
    if (!LanceFieldPathSegmentNeedsQuoting(segment)) {
      out += segment;
      continue;
    }

    out.push_back('`');
    for (auto c : segment) {
      if (c == '`') {
        out += "``";
      } else {
        out.push_back(c);
      }
    }
    out.push_back('`');
  }
  return out;
}

static bool TryBuildLanceColumnIndexPath(const vector<string> &names,
                                         const vector<LogicalType> &types,
                                         const ColumnIndex &col_index,
                                         string &out_path) {
  out_path.clear();
  if (col_index.IsVirtualColumn()) {
    return false;
  }

  auto col_id = col_index.GetPrimaryIndex();
  if (col_id >= names.size() || col_id >= types.size()) {
    return false;
  }

  vector<string> segments;
  segments.push_back(names[col_id]);
  auto leaf_type = types[col_id];

  std::function<bool(const vector<ColumnIndex> &)> append_children;
  append_children = [&](const vector<ColumnIndex> &children) -> bool {
    if (children.empty()) {
      return true;
    }
    if (children.size() != 1) {
      return false;
    }

    auto &child = children[0];
    if (child.IsVirtualColumn()) {
      return false;
    }
    auto child_idx = child.GetPrimaryIndex();
    if (leaf_type.id() != LogicalTypeId::STRUCT ||
        child_idx >= StructType::GetChildCount(leaf_type)) {
      return false;
    }

    segments.push_back(
        StructType::GetChildName(leaf_type, child_idx).GetIdentifierName());
    leaf_type = StructType::GetChildType(leaf_type, child_idx);
    return append_children(child.GetChildIndexes());
  };

  if (!append_children(col_index.GetChildIndexes())) {
    return false;
  }

  out_path = FormatLanceFieldPath(segments);
  return !out_path.empty();
}

static unique_ptr<BaseStatistics>
LanceScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                    column_t column_id) {
  (void)context;
  if (!bind_data_p) {
    return nullptr;
  }
  auto &bind_data = bind_data_p->Cast<LanceScanBindData>();
  if (column_id >= bind_data.types.size()) {
    return nullptr;
  }
  return BaseStatistics::CreateUnknown(bind_data.types[column_id]).ToUnique();
}

static unique_ptr<NodeStatistics>
LanceScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
  (void)context;
  if (!bind_data_p) {
    return nullptr;
  }
  auto &bind_data = bind_data_p->Cast<LanceScanBindData>();

  idx_t count = 0;
  if (!bind_data.take_row_ids.empty()) {
    count = NumericCast<idx_t>(bind_data.take_row_ids.size());
  } else {
    if (!bind_data.dataset) {
      return nullptr;
    }
    auto rows = lance_dataset_count_rows(bind_data.dataset);
    if (rows < 0) {
      return nullptr;
    }
    count = NumericCast<idx_t>(rows);
  }

  if (bind_data.sampling_pushed_down) {
    auto pct = bind_data.sample_percentage / 100.0;
    pct = MaxValue<double>(0.0, MinValue<double>(1.0, pct));
    auto sampled_rows =
        static_cast<int64_t>(std::floor(static_cast<double>(count) * pct));
    count = NumericCast<idx_t>(sampled_rows);
  }

  if (bind_data.limit_offset_pushed_down) {
    if (bind_data.pushed_offset >= count) {
      count = 0;
    } else {
      count -= bind_data.pushed_offset;
      if (bind_data.pushed_limit.IsValid()) {
        count = MinValue<idx_t>(count, bind_data.pushed_limit.GetIndex());
      }
    }
  }

  return make_uniq<NodeStatistics>(count, count);
}

static constexpr column_t LANCE_COLUMN_IDENTIFIER_ROW_ID =
    UINT64_C(9223372036854775900);

static bool LanceExpressionReferencesRowId(const LogicalGet &get,
                                           const Expression &expr) {
  bool found = false;
  std::function<void(const Expression &)> visit = [&](const Expression &node) {
    if (found) {
      return;
    }
    if (node.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
      auto &colref = node.Cast<BoundColumnRefExpression>();
      auto &column_ids = get.GetColumnIds();
      if (colref.Binding().table_index == get.table_index &&
          colref.Binding().column_index < column_ids.size()) {
        auto col_id =
            column_ids[colref.Binding().column_index].GetPrimaryIndex();
        if (col_id == COLUMN_IDENTIFIER_ROW_ID ||
            col_id == LANCE_COLUMN_IDENTIFIER_ROW_ID) {
          found = true;
          return;
        }
      }
    }
    if (node.GetExpressionClass() == ExpressionClass::BOUND_REF) {
      auto ref_index = node.Cast<BoundReferenceExpression>().Index();
      auto &column_ids = get.GetColumnIds();
      if (ref_index < column_ids.size()) {
        auto col_id = column_ids[ref_index].GetPrimaryIndex();
        if (col_id == COLUMN_IDENTIFIER_ROW_ID ||
            col_id == LANCE_COLUMN_IDENTIFIER_ROW_ID) {
          found = true;
          return;
        }
      }
    }
    ExpressionIterator::EnumerateChildren(node, visit);
  };
  visit(expr);
  return found;
}

static bool LancePushdownExpression(ClientContext &, const LogicalGet &get,
                                    Expression &expr) {
  // Keep rowid predicates as expressions. DuckDB's table-filter path drops
  // the virtual rowid binding before physical planning for single-value
  // equality, while LanceRowIdInRewrite can turn the expression directly into
  // a native point lookup.
  if (LanceExpressionReferencesRowId(get, expr)) {
    return false;
  }
  if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
    return true;
  }
  auto &func = expr.Cast<BoundFunctionExpression>();
  auto &name = func.Function().GetName().GetIdentifierName();
  // Keep LIKE/ILIKE as expressions so we can build and surface Lance Filter IR
  // (instead of letting DuckDB rewrite them into TableFilters).
  if (name == "~~" || name == "~~*" || name == "like_escape" ||
      name == "ilike_escape") {
    return false;
  }
  return true;
}

static vector<PartitionStatistics>
LanceScanGetPartitionStats(ClientContext &context,
                           GetPartitionStatsInput &input) {
  (void)context;
  if (!input.bind_data) {
    return {};
  }
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
  PartitionStatistics stats;
  stats.row_start = 0;

  idx_t count = 0;
  if (!bind_data.take_row_ids.empty()) {
    count = NumericCast<idx_t>(bind_data.take_row_ids.size());
  } else {
    if (!bind_data.dataset) {
      return {};
    }
    auto rows = lance_dataset_count_rows(bind_data.dataset);
    if (rows < 0) {
      return {};
    }
    count = NumericCast<idx_t>(rows);
  }

  if (bind_data.sampling_pushed_down) {
    auto pct = bind_data.sample_percentage / 100.0;
    pct = MaxValue<double>(0.0, MinValue<double>(1.0, pct));
    auto sampled_rows =
        static_cast<int64_t>(std::floor(static_cast<double>(count) * pct));
    count = NumericCast<idx_t>(sampled_rows);
  }

  if (bind_data.limit_offset_pushed_down) {
    if (bind_data.pushed_offset >= count) {
      count = 0;
    } else {
      count -= bind_data.pushed_offset;
      if (bind_data.pushed_limit.IsValid()) {
        count = MinValue<idx_t>(count, bind_data.pushed_limit.GetIndex());
      }
    }
  }

  stats.count = count;
  stats.count_type = bind_data.sampling_pushed_down
                         ? CountType::COUNT_APPROXIMATE
                         : CountType::COUNT_EXACT;
  vector<PartitionStatistics> out;
  out.push_back(stats);
  return out;
}

LanceScanBindData::~LanceScanBindData() {}

static constexpr const char *LANCE_ROW_ID_COLUMN_NAME = "_rowid";
static constexpr const char *LANCE_DEFERRED_SETTING =
    "lance_deferred_materialization";
static constexpr const char *LANCE_NAMESPACE_QUERY_TABLE_SETTING =
    "lance_namespace_query_table";
static constexpr uint64_t DEFERRED_AVG_BYTES_THRESHOLD = 1024;

static bool IsLanceVirtualRowIdColumnId(column_t col_id) {
  return col_id == LANCE_COLUMN_IDENTIFIER_ROW_ID;
}

static vector<bool> LanceBlobColumnsFromSchema(const ArrowSchema &schema_root) {
  vector<bool> result;
  result.reserve(schema_root.n_children);
  for (int64_t i = 0; i < schema_root.n_children; i++) {
    result.push_back(schema_root.children && schema_root.children[i] &&
                     LanceArrowFieldContainsBlob(*schema_root.children[i]));
  }
  return result;
}

static bool IsLikelyHeavyColumnType(const LogicalType &type) {
  switch (type.id()) {
  case LogicalTypeId::BLOB:
    return true;
  case LogicalTypeId::LIST: {
    auto &child = ListType::GetChildType(type);
    if (child.IsNumeric() || child.id() == LogicalTypeId::BLOB) {
      return true;
    }
    return IsLikelyHeavyColumnType(child);
  }
  case LogicalTypeId::ARRAY: {
    auto &child = ArrayType::GetChildType(type);
    auto size = ArrayType::GetSize(type);
    if (child.id() == LogicalTypeId::BLOB) {
      return true;
    }
    if (child.IsNumeric() && size >= 256) {
      return true;
    }
    return IsLikelyHeavyColumnType(child);
  }
  case LogicalTypeId::MAP:
    return true;
  case LogicalTypeId::STRUCT: {
    auto &children = StructType::GetChildTypes(type);
    for (auto &entry : children) {
      if (IsLikelyHeavyColumnType(entry.second)) {
        return true;
      }
    }
    return false;
  }
  default:
    return false;
  }
}

static bool LanceDeferredMaterializationEnabled(ClientContext &context) {
  Value val;
  if (context.TryGetCurrentSetting(LANCE_DEFERRED_SETTING, val)) {
    return val.GetValue<bool>();
  }
  return true; // default on
}

// When disabled, REST namespace table scans skip the query_table API and open
// the underlying dataset directly (the pre-query_table behaviour), which is
// required against namespace servers that do not implement query_table.
static bool LanceNamespaceQueryTableEnabled(ClientContext &context) {
  Value val;
  if (context.TryGetCurrentSetting(LANCE_NAMESPACE_QUERY_TABLE_SETTING, val)) {
    return val.GetValue<bool>();
  }
  return true; // default on
}

static virtual_column_map_t LanceGetVirtualColumns(ClientContext &,
                                                   optional_ptr<FunctionData>) {
  virtual_column_map_t result;
  result.emplace(COLUMN_IDENTIFIER_ROW_ID,
                 TableColumn("rowid", LogicalType::ROW_TYPE));
  result.emplace(COLUMN_IDENTIFIER_EMPTY,
                 TableColumn("", LogicalType::BOOLEAN));
  result.emplace(LANCE_COLUMN_IDENTIFIER_ROW_ID,
                 TableColumn(LANCE_ROW_ID_COLUMN_NAME, LogicalType::UBIGINT));
  return result;
}

static bool TryLanceExplainDatasetScan(void *dataset,
                                       const vector<string> *columns,
                                       const string *filter_ir,
                                       const optional_idx &pushed_limit,
                                       idx_t pushed_offset, bool verbose,
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
  idx_t filter_len = 0;
  if (filter_ir && !filter_ir->empty()) {
    filter_ptr = reinterpret_cast<const uint8_t *>(filter_ir->data()); // NOLINT
    filter_len = NumericCast<idx_t>(filter_ir->size());
  }

  auto limit_i64 = pushed_limit.IsValid()
                       ? NumericCast<int64_t>(pushed_limit.GetIndex())
                       : int64_t(-1);
  auto offset_i64 = NumericCast<int64_t>(pushed_offset);

  auto *plan_ptr = lance_explain_dataset_scan_ir(
      dataset, col_ptrs.empty() ? nullptr : col_ptrs.data(), col_ptrs.size(),
      filter_ptr, NumericCast<size_t>(filter_len), limit_i64, offset_i64,
      verbose ? 1 : 0);
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

struct LanceScanGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> next_fragment_idx{0};
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  std::atomic<idx_t> streams_opened{0};
  std::atomic<idx_t> filter_pushdown_fallbacks{0};

  // REST scans choose query_table or native access after the projection is
  // known. Native access pins the dataset and its materialized Arrow schemas.
  shared_ptr<LanceDatasetCacheEntry> native_dataset_entry;
  void *native_dataset = nullptr;
  bool native_dataset_cache_hit = false;
  ArrowSchemaWrapper native_schema_root;
  ArrowSchemaWrapper native_scan_schema_root;
  ArrowTableSchema native_arrow_table;
  ArrowTableSchema native_scan_arrow_table;
  bool use_dataset_scanner = false;
  bool use_namespace_query = false;
  bool sampling_pushed_down = false;
  double sample_percentage = 0.0;
  int64_t sample_seed = -1;
  bool sample_repeatable = false;
  bool use_dataset_take = false;
  bool scan_includes_virtual_rowid = false;
  bool limit_offset_pushed_down = false;
  optional_idx pushed_limit = optional_idx::Invalid();
  idx_t pushed_offset = 0;

  vector<uint64_t> take_row_ids;
  vector<uint64_t> fragment_ids;
  idx_t max_threads = 1;

  vector<idx_t> projection_ids;
  // Types for DuckDB's expected scan columns (input.column_ids order).
  // This can include virtual columns like `rowid` and `_rowid`.
  vector<LogicalType> scanned_types;

  // Types for the Arrow->DuckDB conversion chunk. This is a de-duplicated set
  // of underlying columns that we actually request from Lance.
  vector<LogicalType> scan_converted_types;

  // Mapping from DuckDB scan columns (input.column_ids order) into the
  // conversion chunk.
  vector<idx_t> output_to_scan_converted_idx;
  vector<bool> output_is_duckdb_rowid;
  vector<bool> output_is_empty;
  vector<bool> output_is_forced_null;

  vector<column_t> scan_column_ids;
  vector<string> scan_column_names;
  string lance_filter_ir;
  bool filter_pushed_down = false;

  bool count_only = false;
  idx_t count_only_total_rows = 0;
  std::atomic<idx_t> count_only_offset{0};

  // Deferred materialization for heavy columns when filter pushdown fails.
  bool deferred_materialization = false;
  vector<string> deferred_column_names;
  vector<LogicalType> deferred_column_types;
  vector<idx_t> output_to_deferred_idx;
  idx_t rowid_scan_converted_idx = DConstants::INVALID_INDEX;

  std::atomic<bool> explain_computed{false};
  string explain_plan;
  string explain_error;
  std::mutex explain_mutex;

  idx_t MaxThreads() const override { return max_threads; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

static void *LanceScanDatasetHandle(const LanceScanBindData &bind_data,
                                    const LanceScanGlobalState &global_state) {
  return global_state.native_dataset ? global_state.native_dataset
                                     : bind_data.dataset;
}

struct LanceScanLocalState : public ArrowScanLocalState {
  explicit LanceScanLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                               ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  DataChunk scan_converted;
  DataChunk deferred_converted;
  shared_ptr<ArrowArrayWrapper> deferred_arrow_owner;
  void *stream = nullptr;
  LanceScanGlobalState *global_state = nullptr;
  idx_t fragment_pos = 0;
  idx_t rows_scanned = 0;
  bool filter_pushed_down = false;
  SelectionVector filter_sel;

  ~LanceScanLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

struct LanceExecBindData : public TableFunctionData {
  string file_path;
  string exec_ir;

  shared_ptr<LanceDatasetCacheEntry> dataset_entry;
  void *dataset = nullptr;
  bool dataset_cache_hit = false;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;

  // Pins a dataset handle checked out at bind time; force a rebind per
  // execution so prepared statements observe external commits (see
  // LanceScanBindData::SupportStatementCache).
  bool SupportStatementCache() const override { return false; }
};

static bool LanceSupportsPushdownType(const FunctionData &bind_data,
                                      idx_t col_idx) {
  (void)bind_data;
  (void)col_idx;
  // The scan applies a pushed filter on any type (Lance IR, else
  // ApplyDuckDBFilters); returning false makes DuckDB retain the filter-only
  // column and desync join layout.
  return true;
}

static bool TryParseRowIdValue(const Value &value, uint64_t &out) {
  if (value.IsNull()) {
    return false;
  }

  auto cast_signed = [&out](int64_t v) {
    if (v < 0) {
      return false;
    }
    out = NumericCast<uint64_t>(v);
    return true;
  };

  switch (value.type().id()) {
  case LogicalTypeId::TINYINT:
    return cast_signed(value.GetValue<int8_t>());
  case LogicalTypeId::SMALLINT:
    return cast_signed(value.GetValue<int16_t>());
  case LogicalTypeId::INTEGER:
    return cast_signed(value.GetValue<int32_t>());
  case LogicalTypeId::BIGINT:
    return cast_signed(value.GetValue<int64_t>());
  case LogicalTypeId::UTINYINT:
    out = value.GetValue<uint8_t>();
    return true;
  case LogicalTypeId::USMALLINT:
    out = value.GetValue<uint16_t>();
    return true;
  case LogicalTypeId::UINTEGER:
    out = value.GetValue<uint32_t>();
    return true;
  case LogicalTypeId::UBIGINT:
    out = value.GetValue<uint64_t>();
    return true;
  default:
    return false;
  }
}

static bool TryGetRowIdConstant(const Expression &expr, Value &out) {
  if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
    out = expr.Cast<BoundConstantExpression>().GetValue();
    return true;
  }
  if (!BoundCastExpression::IsCast(expr)) {
    return false;
  }
  auto &cast = expr.Cast<BoundFunctionExpression>();
  if (BoundCastExpression::IsTryCast(cast) || cast.GetChildren().size() != 1 ||
      !cast.GetChildren()[0] ||
      cast.GetChildren()[0]->GetExpressionClass() !=
          ExpressionClass::BOUND_CONSTANT) {
    return false;
  }
  try {
    out = cast.GetChildren()[0]
              ->Cast<BoundConstantExpression>()
              .GetValue()
              .DefaultCastAs(cast.GetReturnType());
    return true;
  } catch (...) {
    return false;
  }
}

static void CastRowIdToDuckDBRowType(Vector &src, Vector &dst, idx_t count) {
  if (count == 0) {
    return;
  }
  if (src.GetType() != LogicalType::UBIGINT ||
      dst.GetType() != LogicalType::ROW_TYPE) {
    throw InternalException("Invalid rowid cast types");
  }

  UnifiedVectorFormat format;
  src.ToUnifiedFormat(format);
  auto src_data = UnifiedVectorFormat::GetData<uint64_t>(format);

  dst.SetVectorType(VectorType::FLAT_VECTOR);
  FlatVector::SetSize(dst, count);
  auto dst_data = FlatVector::GetDataMutable<int64_t>(dst);
  auto &dst_validity = FlatVector::ValidityMutable(dst);
  dst_validity.SetAllValid(count);

  for (idx_t i = 0; i < count; i++) {
    auto idx = format.sel->get_index(i);
    if (!format.validity.RowIsValid(idx)) {
      dst_validity.SetInvalid(i);
      continue;
    }
    auto v = src_data[idx];
    if (v > NumericLimits<int64_t>::Maximum()) {
      throw InvalidInputException(
          "Lance rowid values must fit in signed 64-bit integers");
    }
    dst_data[i] = NumericCast<int64_t>(v);
  }
}

static bool TryExtractTakeRowIdsFromExpression(const Expression &expr,
                                               vector<uint64_t> &out_row_ids) {
  if (auto child = ExpressionFilter::GetOptionalFilterChild(expr)) {
    return TryExtractTakeRowIdsFromExpression(*child, out_row_ids);
  }
  vector<Value> values;
  if (BoundComparisonExpression::IsComparison(expr)) {
    auto &comparison_expr = expr.Cast<BoundFunctionExpression>();
    auto comparison = expr.GetExpressionType();
    if (comparison != ExpressionType::COMPARE_EQUAL ||
        comparison_expr.GetChildren().size() != 2) {
      return false;
    }
    Value value;
    if (TryGetRowIdConstant(*comparison_expr.GetChildren()[1], value)) {
      values.push_back(std::move(value));
    } else if (TryGetRowIdConstant(*comparison_expr.GetChildren()[0], value)) {
      values.push_back(std::move(value));
    } else {
      return false;
    }
  } else if (expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR &&
             expr.GetExpressionType() == ExpressionType::COMPARE_IN) {
    const auto &children = expr.Cast<BoundOperatorExpression>().GetChildren();
    if (children.size() < 2 ||
        !ExpressionFilter::IsSimpleFilterColumnRef(*children[0])) {
      return false;
    }
    for (idx_t i = 1; i < children.size(); i++) {
      Value value;
      if (!TryGetRowIdConstant(*children[i], value)) {
        return false;
      }
      values.push_back(std::move(value));
    }
  } else {
    return false;
  }
  for (const auto &value : values) {
    uint64_t row_id;
    if (!TryParseRowIdValue(value, row_id)) {
      throw InvalidInputException("Lance point lookup requires non-negative "
                                  "integer rowid/_rowid values");
    }
    out_row_ids.push_back(row_id);
  }
  return true;
}

static bool TryExtractTakeRowIdsFromFilter(const TableFilter &filter,
                                           vector<uint64_t> &out_row_ids) {
  const auto &expression_filter =
      ExpressionFilter::GetExpressionFilter(filter, "Lance rowid lookup");
  // The caller has already verified that this filter targets the rowid
  // column. `column_indexes` is not a safe gating signal here: DuckDB's
  // single-value equality filter prunes the rowid projection before physical
  // planning, so a fallback that recreates the logical filter would reference
  // a binding that no longer exists.
  return TryExtractTakeRowIdsFromExpression(*expression_filter.expr,
                                            out_row_ids);
}

static bool IsTakeRowIdFilter(const TableFilter &filter) {
  vector<uint64_t> row_ids;
  return TryExtractTakeRowIdsFromFilter(filter, row_ids);
}

static bool TryExtractTakeRowIdsFromFilters(const TableFilterSet &filters,
                                            const idx_t row_id_col_idx,
                                            vector<uint64_t> &out_row_ids) {
  auto filter =
      filters.TryGetFilterByColumnIndex(ProjectionIndex(row_id_col_idx));
  return filter && TryExtractTakeRowIdsFromFilter(*filter, out_row_ids);
}

static void
LancePushdownComplexFilter(ClientContext &context, LogicalGet &get,
                           FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
  if (!bind_data || filters.empty()) {
    return;
  }
  auto &scan_bind = bind_data->Cast<LanceScanBindData>();

  auto quote_identifier = [](const string &name) {
    string escaped;
    escaped.reserve(name.size() + 2);
    for (auto c : name) {
      if (c == '`') {
        escaped.push_back('`');
        escaped.push_back('`');
      } else {
        escaped.push_back(c);
      }
    }
    return "`" + escaped + "`";
  };

  vector<string> get_column_names;
  auto &col_ids = get.GetColumnIds();
  get_column_names.reserve(col_ids.size());
  for (idx_t i = 0; i < col_ids.size(); i++) {
    auto col_id = col_ids[i].GetPrimaryIndex();
    if (col_id == COLUMN_IDENTIFIER_ROW_ID ||
        col_id == COLUMN_IDENTIFIER_EMPTY) {
      get_column_names.push_back("");
      continue;
    }
    if (col_id >= scan_bind.names.size()) {
      get_column_names.push_back("");
      continue;
    }
    get_column_names.push_back(scan_bind.names[col_id]);
  }

  auto is_rowid_ref = [&](const BoundColumnRefExpression &colref) {
    if (colref.Binding().table_index != get.table_index ||
        colref.Binding().column_index >= col_ids.size()) {
      return false;
    }
    auto col_id = col_ids[colref.Binding().column_index].GetPrimaryIndex();
    return col_id == COLUMN_IDENTIFIER_ROW_ID ||
           IsLanceVirtualRowIdColumnId(col_id);
  };

  std::function<bool(const Expression &, vector<uint64_t> &)>
      try_extract_rowids;
  try_extract_rowids = [&](const Expression &expr, vector<uint64_t> &out) {
    if (expr.GetExpressionType() == ExpressionType::COMPARE_IN &&
        expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
      auto &op = expr.Cast<BoundOperatorExpression>();
      if (op.GetChildren().size() <= 1 || !op.GetChildren()[0] ||
          op.GetChildren()[0]->GetExpressionClass() !=
              ExpressionClass::BOUND_COLUMN_REF) {
        return false;
      }
      auto &colref = op.GetChildren()[0]->Cast<BoundColumnRefExpression>();
      if (!is_rowid_ref(colref)) {
        return false;
      }
      out.clear();
      out.reserve(op.GetChildren().size() - 1);
      for (idx_t i = 1; i < op.GetChildren().size(); i++) {
        auto &child = op.GetChildren()[i];
        Value value;
        if (!child || !TryGetRowIdConstant(*child, value)) {
          return false;
        }
        uint64_t row_id = 0;
        if (!TryParseRowIdValue(value, row_id)) {
          throw InvalidInputException(
              "Lance point lookup requires non-negative integer rowid/_rowid "
              "values");
        }
        out.push_back(row_id);
      }
      return true;
    }

    if (expr.GetExpressionType() == ExpressionType::COMPARE_EQUAL &&
        BoundComparisonExpression::IsComparison(expr)) {
      auto &cmp = expr.Cast<BoundFunctionExpression>();
      if (!cmp.GetChildren()[0] || !cmp.GetChildren()[1]) {
        return false;
      }
      auto extract_one = [&](const Expression &lhs,
                             const Expression &rhs) -> bool {
        if (lhs.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
          return false;
        }
        auto &colref = lhs.Cast<BoundColumnRefExpression>();
        if (!is_rowid_ref(colref)) {
          return false;
        }
        uint64_t row_id = 0;
        Value value;
        if (!TryGetRowIdConstant(rhs, value)) {
          return false;
        }
        if (!TryParseRowIdValue(value, row_id)) {
          throw InvalidInputException(
              "Lance point lookup requires non-negative integer rowid/_rowid "
              "values");
        }
        out.clear();
        out.push_back(row_id);
        return true;
      };
      if (extract_one(*cmp.GetChildren()[0], *cmp.GetChildren()[1])) {
        return true;
      }
      if (extract_one(*cmp.GetChildren()[1], *cmp.GetChildren()[0])) {
        return true;
      }
      return false;
    }

    if (expr.GetExpressionType() == ExpressionType::CONJUNCTION_OR &&
        expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
      auto &conj = expr.Cast<BoundConjunctionExpression>();
      vector<uint64_t> child_out;
      out.clear();
      out.reserve(conj.GetChildren().size());
      for (auto &child : conj.GetChildren()) {
        if (!child) {
          return false;
        }
        child_out.clear();
        if (!try_extract_rowids(*child, child_out) || child_out.empty()) {
          return false;
        }
        out.insert(out.end(), child_out.begin(), child_out.end());
      }
      return !out.empty();
    }

    return false;
  };

  for (auto &expr : filters) {
    if (!expr || expr->HasParameter() || expr->IsVolatile()) {
      continue;
    }
    if (!scan_bind.UsesNamespaceQuery() && scan_bind.take_row_ids.empty()) {
      vector<uint64_t> take_row_ids;
      if (try_extract_rowids(*expr, take_row_ids) && !take_row_ids.empty()) {
        scan_bind.take_row_ids = std::move(take_row_ids);
      }
    }
    if (BoundComparisonExpression::IsComparison(*expr)) {
      auto &cmp = expr->Cast<BoundFunctionExpression>();
      if (cmp.GetExpressionType() == ExpressionType::COMPARE_DISTINCT_FROM ||
          cmp.GetExpressionType() ==
              ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        auto is_constant = [](const unique_ptr<Expression> &node) -> bool {
          if (!node) {
            return false;
          }
          if (node->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
            return true;
          }
          if (BoundCastExpression::IsCast(*node)) {
            auto &cast = node->Cast<BoundFunctionExpression>();
            return !BoundCastExpression::IsTryCast(cast) &&
                   cast.GetChildren()[0] &&
                   cast.GetChildren()[0]->GetExpressionClass() ==
                       ExpressionClass::BOUND_CONSTANT;
          }
          return false;
        };

        auto is_column = [](const unique_ptr<Expression> &node) -> bool {
          if (!node) {
            return false;
          }
          return node->GetExpressionClass() ==
                     ExpressionClass::BOUND_COLUMN_REF ||
                 node->GetExpressionClass() == ExpressionClass::BOUND_REF;
        };

        // Prefer TableFilterSet + ExpressionFilter for the 1-column + constant
        // form (keeps DuckDB fallback available without duplicating the
        // predicate in LancePushedFilterParts).
        if ((is_column(cmp.GetChildren()[0]) &&
             is_constant(cmp.GetChildren()[1])) ||
            (is_column(cmp.GetChildren()[1]) &&
             is_constant(cmp.GetChildren()[0]))) {
          continue;
        }
      }
    }
    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types,
                                   scan_bind.json_columns, false, *expr,
                                   filter_ir)) {
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));

    auto expr_copy = expr->Copy();
    ExpressionIterator::EnumerateExpression(expr_copy, [&](Expression &node) {
      if (node.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return;
      }
      auto &colref = node.Cast<BoundColumnRefExpression>();
      if (colref.Binding().table_index != get.table_index ||
          colref.Binding().column_index >= get_column_names.size()) {
        throw NotImplementedException(
            "Lance scan filter pushdown does not support joins");
      }
      auto &name = get_column_names[colref.Binding().column_index];
      if (name.empty()) {
        throw NotImplementedException(
            "Lance scan filter pushdown could not resolve column name");
      }
      colref.SetAlias(Identifier(quote_identifier(name)));
    });
    auto sql_part = expr_copy->ToString();
    scan_bind.duckdb_pushed_filter_sql_parts.push_back(sql_part);
  }
}

static unique_ptr<FunctionData> LanceScanBind(ClientContext &context,
                                              TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types,
                                              vector<Identifier> &names) {
  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw InvalidInputException("Lance scan requires a dataset root path");
  }

  auto result = make_uniq<LanceScanBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  auto verbose_it = input.named_parameters.find("explain_verbose");
  if (verbose_it != input.named_parameters.end() &&
      !verbose_it->second.IsNull()) {
    result->explain_verbose =
        verbose_it->second.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  result->dataset_entry = LanceGetOrOpenDatasetEntry(
      context, result->file_path, &result->dataset_cache_hit);
  result->dataset =
      result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
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
  auto schema_coercion =
      LanceCoerceArrowSchemaForDuckDB(&result->schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  result->json_columns = std::move(schema_coercion.json_columns);
  result->blob_columns =
      LanceBlobColumnsFromSchema(result->schema_root.arrow_schema);

  auto *scan_schema_handle = lance_get_schema_for_scan(result->dataset);
  if (!scan_schema_handle) {
    throw IOException("Failed to get scan schema from Lance dataset: " +
                      result->file_path + LanceFormatErrorSuffix());
  }
  memset(&result->scan_schema_root.arrow_schema, 0,
         sizeof(result->scan_schema_root.arrow_schema));
  if (lance_schema_to_arrow(scan_schema_handle,
                            &result->scan_schema_root.arrow_schema) != 0) {
    lance_free_schema(scan_schema_handle);
    throw IOException(
        "Failed to export Lance scan schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(scan_schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&result->scan_schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->scan_arrow_table, result->scan_schema_root.arrow_schema);
  names.clear();
  for (const auto &name : result->names) {
    names.emplace_back(name);
  }
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<FunctionData>
LanceNamespaceScanBind(ClientContext &context, TableFunctionBindInput &input,
                       vector<LogicalType> &return_types,
                       vector<Identifier> &names) {
  if (input.inputs.size() < 2 || input.inputs[0].IsNull() ||
      input.inputs[1].IsNull()) {
    throw InvalidInputException(
        "Lance namespace scan requires (endpoint, table_id[, delimiter])");
  }

  auto endpoint =
      input.inputs[0].DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
  auto table_id =
      input.inputs[1].DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
  string delimiter;
  if (input.inputs.size() >= 3 && !input.inputs[2].IsNull()) {
    delimiter =
        input.inputs[2].DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
  }

  auto result = make_uniq<LanceScanBindData>();
  result->file_path = endpoint + "/" + table_id;

  auto verbose_it = input.named_parameters.find("explain_verbose");
  if (verbose_it != input.named_parameters.end() &&
      !verbose_it->second.IsNull()) {
    result->explain_verbose =
        verbose_it->second.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  string bearer_token;
  string api_key;
  ResolveLanceNamespaceAuth(context, endpoint, input.named_parameters,
                            bearer_token, api_key);

  string table_uri;
  string headers_tsv; // TODO: Add support for headers in table function
  result->dataset_entry = LanceGetOrOpenDatasetEntryInNamespace(
      context, endpoint, table_id, bearer_token, api_key, delimiter,
      headers_tsv, table_uri, &result->dataset_cache_hit);
  result->dataset =
      result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
  if (!table_uri.empty()) {
    result->file_path = table_uri;
  }
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset via namespace: " +
                      result->file_path + LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_schema(result->dataset);
  if (!schema_handle) {
    throw IOException(
        "Failed to get schema from Lance dataset via namespace: " +
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
  auto schema_coercion =
      LanceCoerceArrowSchemaForDuckDB(&result->schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  result->json_columns = std::move(schema_coercion.json_columns);
  result->blob_columns =
      LanceBlobColumnsFromSchema(result->schema_root.arrow_schema);

  auto *scan_schema_handle = lance_get_schema_for_scan(result->dataset);
  if (!scan_schema_handle) {
    throw IOException(
        "Failed to get scan schema from Lance dataset via namespace: " +
        result->file_path + LanceFormatErrorSuffix());
  }
  memset(&result->scan_schema_root.arrow_schema, 0,
         sizeof(result->scan_schema_root.arrow_schema));
  if (lance_schema_to_arrow(scan_schema_handle,
                            &result->scan_schema_root.arrow_schema) != 0) {
    lance_free_schema(scan_schema_handle);
    throw IOException(
        "Failed to export Lance scan schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(scan_schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&result->scan_schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->scan_arrow_table, result->scan_schema_root.arrow_schema);
  names.clear();
  for (const auto &name : result->names) {
    names.emplace_back(name);
  }
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
  auto state = make_uniq_base<GlobalTableFunctionState, LanceScanGlobalState>();
  auto &scan_state = state->Cast<LanceScanGlobalState>();

  scan_state.limit_offset_pushed_down = bind_data.limit_offset_pushed_down;
  scan_state.pushed_limit = bind_data.pushed_limit;
  scan_state.pushed_offset = bind_data.pushed_offset;

  scan_state.projection_ids = input.projection_ids;
  auto rowid_internal_index = NumericCast<column_t>(bind_data.types.size());

  scan_state.scanned_types.reserve(input.column_ids.size());
  scan_state.output_to_scan_converted_idx.reserve(input.column_ids.size());
  scan_state.output_is_duckdb_rowid.reserve(input.column_ids.size());
  scan_state.output_is_empty.reserve(input.column_ids.size());
  scan_state.output_is_forced_null.reserve(input.column_ids.size());

  unordered_map<column_t, idx_t> scan_idx_by_col_id;
  scan_idx_by_col_id.reserve(input.column_ids.size());
  auto add_scan_column = [&](column_t col_id, const string &name,
                             const LogicalType &type) -> idx_t {
    auto it = scan_idx_by_col_id.find(col_id);
    if (it != scan_idx_by_col_id.end()) {
      return it->second;
    }
    auto idx = scan_state.scan_column_ids.size();
    scan_state.scan_column_ids.push_back(col_id);
    scan_state.scan_column_names.push_back(name);
    scan_state.scan_converted_types.push_back(type);
    scan_idx_by_col_id.emplace(col_id, idx);
    return idx;
  };

  bool has_projected_blob = false;

  for (auto col_id : input.column_ids) {
    if (col_id == COLUMN_IDENTIFIER_EMPTY) {
      scan_state.scanned_types.push_back(LogicalType::BOOLEAN);
      scan_state.output_to_scan_converted_idx.push_back(
          DConstants::INVALID_INDEX);
      scan_state.output_is_duckdb_rowid.push_back(false);
      scan_state.output_is_empty.push_back(true);
      scan_state.output_is_forced_null.push_back(false);
      continue;
    }

    if (col_id == COLUMN_IDENTIFIER_ROW_ID) {
      scan_state.scan_includes_virtual_rowid = true;
      auto scan_idx = add_scan_column(
          rowid_internal_index, LANCE_ROW_ID_COLUMN_NAME, LogicalType::UBIGINT);
      scan_state.scanned_types.push_back(LogicalType::ROW_TYPE);
      scan_state.output_to_scan_converted_idx.push_back(scan_idx);
      scan_state.output_is_duckdb_rowid.push_back(true);
      scan_state.output_is_empty.push_back(false);
      scan_state.output_is_forced_null.push_back(false);
      continue;
    }

    if (IsLanceVirtualRowIdColumnId(col_id)) {
      scan_state.scan_includes_virtual_rowid = true;
      auto scan_idx = add_scan_column(
          rowid_internal_index, LANCE_ROW_ID_COLUMN_NAME, LogicalType::UBIGINT);
      scan_state.scanned_types.push_back(LogicalType::UBIGINT);
      scan_state.output_to_scan_converted_idx.push_back(scan_idx);
      scan_state.output_is_duckdb_rowid.push_back(false);
      scan_state.output_is_empty.push_back(false);
      scan_state.output_is_forced_null.push_back(false);
      continue;
    }

    if (col_id >= bind_data.types.size() || col_id >= bind_data.names.size()) {
      throw IOException("Invalid column id in projection");
    }
    if (bind_data.forced_null_columns.count(col_id)) {
      scan_state.scanned_types.push_back(bind_data.types[col_id]);
      scan_state.output_to_scan_converted_idx.push_back(
          DConstants::INVALID_INDEX);
      scan_state.output_is_duckdb_rowid.push_back(false);
      scan_state.output_is_empty.push_back(false);
      scan_state.output_is_forced_null.push_back(true);
      continue;
    }
    auto scan_idx = add_scan_column(col_id, bind_data.names[col_id],
                                    bind_data.types[col_id]);
    if (col_id < bind_data.blob_columns.size() &&
        bind_data.blob_columns[col_id]) {
      has_projected_blob = true;
    }
    scan_state.scanned_types.push_back(bind_data.types[col_id]);
    scan_state.output_to_scan_converted_idx.push_back(scan_idx);
    scan_state.output_is_duckdb_rowid.push_back(false);
    scan_state.output_is_empty.push_back(false);
    scan_state.output_is_forced_null.push_back(false);
  }

  shared_ptr<LanceDatasetCacheEntry> write_dataset;
  if (bind_data.UsesNamespaceQuery() && bind_data.table_entry) {
    write_dataset = LanceGetDatasetPinnedForWrite(
        context, bind_data.table_entry->Cast<LanceTableEntry>());
  }

  // Blob projection and a write's statement-pinned dataset force native access,
  // but the bind schema stays unchanged. Decode with the native Arrow schema.
  auto open_native_dataset = [&]() {
    if (scan_state.native_dataset) {
      return;
    }
    if (!bind_data.table_entry) {
      throw InternalException(
          "Lance namespace query scan is missing its table entry");
    }
    auto &table = bind_data.table_entry.get_mutable()->Cast<LanceTableEntry>();
    string display_uri;
    scan_state.native_dataset_entry = write_dataset;
    if (!scan_state.native_dataset_entry) {
      scan_state.native_dataset_entry = LanceGetOrOpenDatasetEntryForTable(
          context, table, display_uri, &scan_state.native_dataset_cache_hit);
    } else {
      display_uri = scan_state.native_dataset_entry->DisplayUri();
    }
    scan_state.native_dataset = scan_state.native_dataset_entry
                                    ? scan_state.native_dataset_entry->Handle()
                                    : nullptr;
    if (!scan_state.native_dataset) {
      throw IOException("Failed to open Lance dataset: " + display_uri +
                        LanceFormatErrorSuffix());
    }
    table.VerifySchemaFreshness(context);

    auto *schema_handle = lance_get_schema(scan_state.native_dataset);
    if (!schema_handle) {
      throw IOException("Failed to get schema from Lance dataset: " +
                        display_uri + LanceFormatErrorSuffix());
    }
    scan_state.native_schema_root.arrow_schema = {};
    if (lance_schema_to_arrow(
            schema_handle, &scan_state.native_schema_root.arrow_schema) != 0) {
      lance_free_schema(schema_handle);
      throw IOException(
          "Failed to export Lance schema to Arrow C Data Interface" +
          LanceFormatErrorSuffix());
    }
    lance_free_schema(schema_handle);
    LanceCoerceArrowSchemaForDuckDB(
        &scan_state.native_schema_root.arrow_schema);
    ArrowTableFunction::PopulateArrowTableSchema(
        context, scan_state.native_arrow_table,
        scan_state.native_schema_root.arrow_schema);

    auto *scan_schema_handle =
        lance_get_schema_for_scan(scan_state.native_dataset);
    if (!scan_schema_handle) {
      throw IOException("Failed to get scan schema from Lance dataset: " +
                        display_uri + LanceFormatErrorSuffix());
    }
    scan_state.native_scan_schema_root.arrow_schema = {};
    if (lance_schema_to_arrow(
            scan_schema_handle,
            &scan_state.native_scan_schema_root.arrow_schema) != 0) {
      lance_free_schema(scan_schema_handle);
      throw IOException(
          "Failed to export Lance scan schema to Arrow C Data Interface" +
          LanceFormatErrorSuffix());
    }
    lance_free_schema(scan_schema_handle);
    LanceCoerceArrowSchemaForDuckDB(
        &scan_state.native_scan_schema_root.arrow_schema);
    ArrowTableFunction::PopulateArrowTableSchema(
        context, scan_state.native_scan_arrow_table,
        scan_state.native_scan_schema_root.arrow_schema);
  };
  if (bind_data.UsesNamespaceQuery() && (has_projected_blob || write_dataset)) {
    open_native_dataset();
  }

  if (input.sample_options &&
      input.sample_options->method == SampleMethod::SYSTEM_SAMPLE &&
      input.sample_options->is_percentage) {
    scan_state.sampling_pushed_down = true;
    scan_state.sample_percentage =
        input.sample_options->sample_size.DefaultCastAs(LogicalType::DOUBLE)
            .GetValue<double>();
    scan_state.sample_seed = input.sample_options->GetSeed();
    scan_state.sample_repeatable = input.sample_options->repeatable;

    // Sampling is applied before filtering and limiting, so do not combine it
    // with the Lance-side filter / limit-offset pushdown paths.
    scan_state.limit_offset_pushed_down = false;
    scan_state.pushed_limit = optional_idx::Invalid();
    scan_state.pushed_offset = 0;

    scan_state.filter_pushed_down = false;
    scan_state.lance_filter_ir.clear();

    scan_state.use_dataset_scanner = true;
    scan_state.max_threads = 1;
  } else {
    vector<string> filter_parts;

    auto table_filters = BuildLanceTableFilterIRParts(
        bind_data.names, bind_data.types, bind_data.json_columns, input, false);
    filter_parts = std::move(table_filters.parts);

    if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
      filter_parts.reserve(filter_parts.size() +
                           bind_data.lance_pushed_filter_ir_parts.size());
      for (auto &part : bind_data.lance_pushed_filter_ir_parts) {
        filter_parts.push_back(part);
      }
    }

    string filter_ir_msg;
    if (!filter_parts.empty() &&
        TryEncodeLanceFilterIRMessage(filter_parts, filter_ir_msg)) {
      scan_state.lance_filter_ir = std::move(filter_ir_msg);
    }
    scan_state.filter_pushed_down =
        table_filters.all_filters_pushed && !scan_state.lance_filter_ir.empty();
  }

  if (bind_data.UsesNamespaceQuery() && !scan_state.native_dataset) {
    if (scan_state.sampling_pushed_down) {
      throw InternalException(
          "Lance namespace query_table scan does not support sampling");
    }

    if (has_projected_blob) {
      // Blob projections must use the native dataset path; query_table would
      // return Lance's physical descriptor structs.
      throw InternalException(
          "Namespace query_table scan cannot materialize Lance blob columns");
    } else {
      scan_state.use_namespace_query = true;
      scan_state.use_dataset_scanner = true;
      scan_state.max_threads = 1;
      bool namespace_query_applicable = true;

      if (scan_state.pushed_limit.IsValid() &&
          scan_state.pushed_limit.GetIndex() == 0) {
        scan_state.count_only = true;
        scan_state.count_only_total_rows = 0;
        return state;
      }

      // query_table treats an omitted projection as "all columns". Request one
      // physical column for count(*) and rowid-only scans so the response has
      // a stable, minimal schema while preserving its row count.
      bool has_physical_column = false;
      for (auto col_id : scan_state.scan_column_ids) {
        if (col_id < bind_data.types.size()) {
          has_physical_column = true;
          break;
        }
      }
      if (!has_physical_column && !bind_data.names.empty()) {
        idx_t driver = bind_data.names.size();
        for (idx_t col_idx = 0; col_idx < bind_data.names.size(); col_idx++) {
          if (col_idx >= bind_data.blob_columns.size() ||
              !bind_data.blob_columns[col_idx]) {
            driver = col_idx;
            break;
          }
        }
        if (driver == bind_data.names.size()) {
          // Blob-only tables cannot express an empty query_table projection.
          // Fall back to the native dataset.
          namespace_query_applicable = false;
          scan_state.use_namespace_query = false;
          scan_state.use_dataset_scanner = true;
          open_native_dataset();
        } else {
          add_scan_column(driver, bind_data.names[driver],
                          bind_data.types[driver]);
        }
      }
      if (namespace_query_applicable) {
        return state;
      }
    }
  }

  auto *dataset = LanceScanDatasetHandle(bind_data, scan_state);

  if (!bind_data.take_row_ids.empty()) {
    if (scan_state.sampling_pushed_down) {
      // Sampling is applied before filtering, so do not combine it with point
      // lookup pushdown (which would filter first).
      return state;
    }
    if (bind_data.limit_offset_pushed_down) {
      throw IOException(
          "Lance point lookup does not support limit/offset pushdown");
    }
    scan_state.lance_filter_ir.clear();
    scan_state.filter_pushed_down = false;
    scan_state.use_dataset_scanner = true;
    scan_state.use_dataset_take = true;
    scan_state.max_threads = 1;
    scan_state.take_row_ids = bind_data.take_row_ids;
    return state;
  }

  idx_t row_id_col_idx = DConstants::INVALID_INDEX;
  for (idx_t i = 0; i < bind_data.names.size(); i++) {
    if (bind_data.names[i] == "_rowid") {
      row_id_col_idx = i;
      break;
    }
  }
  if (row_id_col_idx != DConstants::INVALID_INDEX && input.filters &&
      TryExtractTakeRowIdsFromFilters(*input.filters, row_id_col_idx,
                                      scan_state.take_row_ids)) {
    if (scan_state.sampling_pushed_down) {
      // Sampling is applied before filtering, so do not combine it with point
      // lookup pushdown (which would filter first).
      scan_state.take_row_ids.clear();
    } else {
      if (bind_data.limit_offset_pushed_down) {
        throw IOException(
            "Lance point lookup does not support limit/offset pushdown");
      }
      scan_state.lance_filter_ir.clear();
      scan_state.filter_pushed_down = false;
      scan_state.use_dataset_scanner = true;
      scan_state.use_dataset_take = true;
      scan_state.max_threads = 1;
      return state;
    }
  }

  if (scan_state.scan_column_names.empty() &&
      (!input.filters || !(input.filters->HasFilters() ||
                           input.filters->HasMultiColumnFilters())) &&
      scan_state.lance_filter_ir.empty()) {
    auto rows = lance_dataset_count_rows(dataset);
    if (rows < 0) {
      throw IOException("Failed to count Lance rows" +
                        LanceFormatErrorSuffix());
    }
    if (scan_state.sampling_pushed_down) {
      auto pct = scan_state.sample_percentage / 100.0;
      pct = MaxValue<double>(0.0, MinValue<double>(1.0, pct));
      rows = static_cast<int64_t>(std::floor(static_cast<double>(rows) * pct));
    }
    scan_state.count_only = true;
    auto total_rows = NumericCast<idx_t>(rows);
    if (scan_state.limit_offset_pushed_down) {
      if (scan_state.pushed_offset >= total_rows) {
        scan_state.count_only_total_rows = 0;
      } else {
        auto remaining = total_rows - scan_state.pushed_offset;
        if (scan_state.pushed_limit.IsValid()) {
          scan_state.count_only_total_rows =
              MinValue<idx_t>(remaining, scan_state.pushed_limit.GetIndex());
        } else {
          scan_state.count_only_total_rows = remaining;
        }
      }
    } else {
      scan_state.count_only_total_rows = total_rows;
    }
    scan_state.max_threads = 1;
    return state;
  }

  // -- Deferred materialization for heavy columns --
  // Activates when DuckDB has filters that were NOT fully pushed to Lance,
  // meaning DuckDB will filter post-scan. We defer heavy columns to avoid
  // reading them for rows that will be filtered out.
  // (When filter IS pushed, Lance SDK handles late materialization internally.)
  if (!scan_state.sampling_pushed_down && !scan_state.filter_pushed_down &&
      !scan_state.scan_column_names.empty() && input.filters &&
      (input.filters->HasFilters() || input.filters->HasMultiColumnFilters()) &&
      LanceDeferredMaterializationEnabled(context)) {
    // Collect columns referenced by DuckDB-side filters.
    unordered_set<string> filter_referenced_columns;
    for (auto &it : *input.filters) {
      auto scan_col_idx = it.GetIndex();
      if (scan_col_idx < input.column_ids.size()) {
        auto col_id = input.column_ids[scan_col_idx];
        if (col_id < bind_data.names.size()) {
          filter_referenced_columns.insert(bind_data.names[col_id]);
        }
      }
    }

    for (const auto &filter : input.filters->GetMultiColumnFilters()) {
      const auto &expression_filter = ExpressionFilter::GetExpressionFilter(
          *filter, "Lance deferred materialization");
      for (auto projection : expression_filter.column_indexes) {
        if (projection < input.column_ids.size() &&
            input.column_ids[projection] < bind_data.names.size()) {
          filter_referenced_columns.insert(
              bind_data.names[input.column_ids[projection]]);
        }
      }
    }

    // Detect heavy columns: stats-based with type-based fallback.
    unordered_set<string> heavy_columns;
    auto total_rows = lance_dataset_count_rows(dataset);
    if (total_rows > 0) {
      size_t stats_len = 0;
      auto stats = lance_dataset_list_named_field_stats(dataset, &stats_len);
      unordered_set<string> stats_covered;
      if (stats) {
        for (size_t i = 0; i < stats_len; i++) {
          if (!stats[i].name) {
            continue;
          }
          string col_name = stats[i].name;
          stats_covered.insert(col_name);
          auto avg_bytes =
              stats[i].bytes_on_disk / static_cast<uint64_t>(total_rows);
          if (avg_bytes > DEFERRED_AVG_BYTES_THRESHOLD) {
            heavy_columns.insert(col_name);
          }
        }
        lance_free_named_field_stats_list(stats, stats_len);
      }
      for (idx_t ci = 0; ci < scan_state.scan_column_names.size(); ci++) {
        auto &name = scan_state.scan_column_names[ci];
        bool source_blob_column =
            ci < scan_state.scan_column_ids.size() &&
            scan_state.scan_column_ids[ci] < bind_data.blob_columns.size() &&
            bind_data.blob_columns[scan_state.scan_column_ids[ci]];
        if (source_blob_column ||
            (stats_covered.count(name) == 0 &&
             IsLikelyHeavyColumnType(scan_state.scan_converted_types[ci]))) {
          heavy_columns.insert(name);
        }
      }
    }

    // Never defer filter-referenced or _rowid columns.
    for (auto &name : filter_referenced_columns) {
      heavy_columns.erase(name);
    }
    heavy_columns.erase(LANCE_ROW_ID_COLUMN_NAME);

    // Identify heavy scan column indices.
    unordered_set<idx_t> heavy_scan_indices;
    for (idx_t i = 0; i < scan_state.scan_column_names.size(); i++) {
      if (heavy_columns.count(scan_state.scan_column_names[i])) {
        heavy_scan_indices.insert(i);
      }
    }

    if (!heavy_scan_indices.empty()) {
      scan_state.deferred_materialization = true;

      vector<column_t> new_scan_column_ids;
      vector<string> new_scan_column_names;
      vector<LogicalType> new_scan_converted_types;
      unordered_map<idx_t, idx_t> old_to_new_scan_idx;
      unordered_map<idx_t, idx_t> old_to_deferred_idx;

      for (idx_t i = 0; i < scan_state.scan_column_names.size(); i++) {
        if (heavy_scan_indices.count(i)) {
          auto deferred_idx = scan_state.deferred_column_names.size();
          old_to_deferred_idx[i] = deferred_idx;
          scan_state.deferred_column_names.push_back(
              scan_state.scan_column_names[i]);
          scan_state.deferred_column_types.push_back(
              scan_state.scan_converted_types[i]);
          continue;
        }
        auto new_idx = new_scan_column_names.size();
        old_to_new_scan_idx[i] = new_idx;
        new_scan_column_ids.push_back(scan_state.scan_column_ids[i]);
        new_scan_column_names.push_back(scan_state.scan_column_names[i]);
        new_scan_converted_types.push_back(scan_state.scan_converted_types[i]);
      }

      // Ensure _rowid is in the light scan.
      bool has_rowid = false;
      for (idx_t i = 0; i < new_scan_column_names.size(); i++) {
        if (new_scan_column_names[i] == LANCE_ROW_ID_COLUMN_NAME) {
          has_rowid = true;
          scan_state.rowid_scan_converted_idx = i;
          break;
        }
      }
      if (!has_rowid) {
        scan_state.rowid_scan_converted_idx = new_scan_column_names.size();
        auto rowid_col_id = NumericCast<column_t>(bind_data.types.size());
        new_scan_column_ids.push_back(rowid_col_id);
        new_scan_column_names.push_back(LANCE_ROW_ID_COLUMN_NAME);
        new_scan_converted_types.push_back(LogicalType::UBIGINT);
        scan_state.scan_includes_virtual_rowid = true;
      }

      scan_state.scan_column_ids = std::move(new_scan_column_ids);
      scan_state.scan_column_names = std::move(new_scan_column_names);
      scan_state.scan_converted_types = std::move(new_scan_converted_types);

      // Update output→scan and output→deferred mappings.
      scan_state.output_to_deferred_idx.resize(
          scan_state.output_to_scan_converted_idx.size(),
          DConstants::INVALID_INDEX);
      for (idx_t i = 0; i < scan_state.output_to_scan_converted_idx.size();
           i++) {
        auto old_idx = scan_state.output_to_scan_converted_idx[i];
        if (old_idx == DConstants::INVALID_INDEX) {
          continue;
        }
        auto it_new = old_to_new_scan_idx.find(old_idx);
        if (it_new != old_to_new_scan_idx.end()) {
          scan_state.output_to_scan_converted_idx[i] = it_new->second;
        } else {
          scan_state.output_to_scan_converted_idx[i] =
              DConstants::INVALID_INDEX;
          auto it_def = old_to_deferred_idx.find(old_idx);
          if (it_def != old_to_deferred_idx.end()) {
            scan_state.output_to_deferred_idx[i] = it_def->second;
          }
        }
      }
    }
  }

  if (!scan_state.sampling_pushed_down && bind_data.limit_offset_pushed_down) {
    // Limit/offset pushdown requires that any TableFilterSet predicates are
    // evaluated by Lance. Otherwise limit/offset would apply before filtering.
    // Filter pushability is pre-checked in LanceLimitOffsetPushdown (the
    // optimizer pass that sets limit_offset_pushed_down), so reaching here
    // with a non-pushable filter set indicates an optimizer bug.
    if (input.filters &&
        (input.filters->HasFilters() ||
         input.filters->HasMultiColumnFilters()) &&
        !scan_state.filter_pushed_down) {
      throw IOException("Lance limit/offset pushdown requires filter pushdown");
    }
    scan_state.use_dataset_scanner = true;
    scan_state.max_threads = 1;
    return state;
  }

  if (scan_state.sampling_pushed_down) {
    scan_state.use_dataset_scanner = true;
    scan_state.max_threads = 1;
    return state;
  }

  // Index-aware scanner selection: if any filtered column has a scalar index,
  // use the dataset scanner so Lance can leverage the index (e.g. BTree skip).
  if (!scan_state.lance_filter_ir.empty()) {
    unordered_set<string> filtered_columns;
    if (input.filters) {
      for (auto &it : *input.filters) {
        auto scan_col_idx = it.GetIndex();
        if (scan_col_idx < input.column_ids.size()) {
          string filtered_path;
          if (scan_col_idx < input.column_indexes.size() &&
              TryBuildLanceColumnIndexPath(bind_data.names, bind_data.types,
                                           input.column_indexes[scan_col_idx],
                                           filtered_path)) {
            filtered_columns.insert(std::move(filtered_path));
          } else {
            auto col_id = input.column_ids[scan_col_idx];
            if (col_id < bind_data.names.size()) {
              filtered_columns.insert(bind_data.names[col_id]);
            }
          }
        }
      }
    }

    if (!filtered_columns.empty()) {
      size_t indexed_cols_len = 0;
      auto indexed_cols_ptr =
          lance_dataset_list_scalar_indexed_columns(dataset, &indexed_cols_len);
      bool has_indexed_filter = false;
      for (size_t i = 0; i < indexed_cols_len; i++) {
        if (indexed_cols_ptr[i] &&
            filtered_columns.count(indexed_cols_ptr[i])) {
          has_indexed_filter = true;
          break;
        }
      }
      lance_free_scalar_indexed_columns(indexed_cols_ptr, indexed_cols_len);
      if (has_indexed_filter) {
        scan_state.use_dataset_scanner = true;
        scan_state.max_threads = 1;
        return state;
      }
    }
  }

  size_t ffi_fragment_count = 0;
  auto fragments_ptr =
      lance_dataset_list_fragments(dataset, &ffi_fragment_count);
  if (!fragments_ptr) {
    throw IOException("Failed to list Lance fragments" +
                      LanceFormatErrorSuffix());
  }
  auto fragment_count = NumericCast<idx_t>(ffi_fragment_count);
  scan_state.fragment_ids.assign(
      fragments_ptr, fragments_ptr + NumericCast<size_t>(fragment_count));
  lance_free_fragment_list(fragments_ptr, ffi_fragment_count);

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
  result->column_ids = scan_global.scan_column_ids;
  result->filters = input.filters.get();
  result->global_state = &scan_global;
  result->filter_pushed_down = scan_global.filter_pushed_down;
  if (scan_global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, scan_global.scanned_types);
  }
  result->scan_converted.Initialize(context.client,
                                    scan_global.scan_converted_types);
  if (scan_global.deferred_materialization) {
    result->deferred_converted.Initialize(context.client,
                                          scan_global.deferred_column_types);
  }
  if (scan_global.count_only) {
    return std::move(result);
  }
  if (scan_global.use_dataset_scanner) {
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

  auto *dataset = LanceScanDatasetHandle(bind_data, global_state);
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
  local_state.filter_pushed_down =
      global_state.filter_pushed_down && filter_ir && filter_ir_len > 0;

  void *stream = nullptr;
  if (global_state.use_namespace_query) {
    if (!bind_data.namespace_query_config) {
      throw InternalException(
          "Lance namespace query scan is missing its namespace config");
    }

    vector<const char *> option_key_ptrs;
    vector<const char *> option_value_ptrs;
    vector<const char *> namespace_column_ptrs;
    string bearer_token;
    string api_key;
    LanceNamespaceQueryConfig config{};
    FillLanceNamespaceQueryConfig(
        context, *bind_data.namespace_query_config, 0, true, "",
        global_state.scan_column_names, option_key_ptrs, option_value_ptrs,
        namespace_column_ptrs, bearer_token, api_key, config);

    auto limit_i64 =
        global_state.pushed_limit.IsValid()
            ? NumericCast<int64_t>(global_state.pushed_limit.GetIndex())
            : int64_t(-1);
    auto offset_i64 = NumericCast<int64_t>(global_state.pushed_offset);
    stream = lance_create_namespace_scan_stream_ir(
        &config, filter_ir, filter_ir_len, limit_i64, offset_i64,
        global_state.scan_includes_virtual_rowid ? 1 : 0);
    if (!stream && filter_ir) {
      if (global_state.limit_offset_pushed_down) {
        throw IOException("Lance namespace query_table filter pushdown failed" +
                          LanceFormatErrorSuffix());
      }
      // Without a pushed LIMIT, retrying without the remote filter preserves
      // correctness because DuckDB still evaluates the predicate locally.
      global_state.filter_pushdown_fallbacks.fetch_add(1);
      local_state.filter_pushed_down = false;
      stream = lance_create_namespace_scan_stream_ir(
          &config, nullptr, 0, limit_i64, offset_i64,
          global_state.scan_includes_virtual_rowid ? 1 : 0);
    }
  } else if (global_state.sampling_pushed_down) {
    local_state.filter_pushed_down = false;
    stream = lance_create_dataset_sample_stream_ir(
        dataset, columns.data(), columns.size(), global_state.sample_percentage,
        global_state.sample_seed, global_state.sample_repeatable ? 1 : 0);
  } else if (global_state.use_dataset_take) {
    auto row_ids_ptr = global_state.take_row_ids.empty()
                           ? nullptr
                           : global_state.take_row_ids.data();
    stream = lance_create_dataset_take_stream(dataset, row_ids_ptr,
                                              global_state.take_row_ids.size(),
                                              columns.data(), columns.size());
  } else if (global_state.use_dataset_scanner) {
    auto limit_i64 =
        global_state.pushed_limit.IsValid()
            ? NumericCast<int64_t>(global_state.pushed_limit.GetIndex())
            : int64_t(-1);
    auto offset_i64 = NumericCast<int64_t>(global_state.pushed_offset);
    stream = lance_create_dataset_stream_ir(
        dataset, columns.data(), columns.size(), filter_ir, filter_ir_len,
        limit_i64, offset_i64);
    if (!stream && filter_ir) {
      if (global_state.limit_offset_pushed_down &&
          local_state.filter_pushed_down) {
        throw IOException("Lance dataset scan filter pushdown failed" +
                          LanceFormatErrorSuffix());
      }
      // Best-effort: if filter pushdown failed, retry without it and rely on
      // DuckDB-side filter execution for correctness.
      global_state.filter_pushdown_fallbacks.fetch_add(1);
      local_state.filter_pushed_down = false;
      stream = lance_create_dataset_stream_ir(dataset, columns.data(),
                                              columns.size(), nullptr, 0,
                                              limit_i64, offset_i64);
    }
  } else {
    if (local_state.fragment_pos >= global_state.fragment_ids.size()) {
      return false;
    }
    auto fragment_id = global_state.fragment_ids[local_state.fragment_pos];

    stream = lance_create_fragment_stream_ir(dataset, fragment_id,
                                             columns.data(), columns.size(),
                                             filter_ir, filter_ir_len);
    if (!stream && filter_ir) {
      // Best-effort: if filter pushdown failed, retry without it and rely on
      // DuckDB-side filter execution for correctness.
      global_state.filter_pushdown_fallbacks.fetch_add(1);
      local_state.filter_pushed_down = false;
      stream = lance_create_fragment_stream_ir(
          dataset, fragment_id, columns.data(), columns.size(), nullptr, 0);
    }
  }
  if (!stream) {
    auto message = global_state.use_namespace_query
                       ? "Failed to create Lance namespace query_table stream"
                       : "Failed to create Lance scan stream";
    throw IOException(message + LanceFormatErrorSuffix());
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

  LanceArrowBatch exported(batch);
  auto &new_chunk = exported.array;
  auto &tmp_schema = exported.schema.arrow_schema;

  if (local_state.global_state && tmp_schema.n_children > 0 &&
      new_chunk->arrow_array.n_children == tmp_schema.n_children &&
      !local_state.global_state->scan_column_names.empty()) {
    unordered_map<string, idx_t> idx_by_name;
    idx_t child_count = NumericCast<idx_t>(tmp_schema.n_children);
    idx_by_name.reserve(child_count);
    for (idx_t i = 0; i < child_count; i++) {
      auto *child_schema = tmp_schema.children[i];
      if (!child_schema || !child_schema->name) {
        continue;
      }
      idx_by_name.emplace(child_schema->name, i);
    }

    auto expected_count =
        NumericCast<idx_t>(local_state.global_state->scan_column_names.size());
    vector<ArrowArray *> old_children;
    old_children.reserve(child_count);
    for (idx_t i = 0; i < child_count; i++) {
      old_children.push_back(new_chunk->arrow_array.children[i]);
    }

    for (idx_t i = 0; i < expected_count; i++) {
      auto &expected = local_state.global_state->scan_column_names[i];
      auto it = idx_by_name.find(expected);
      if (it == idx_by_name.end()) {
        throw IOException("Missing expected column in Arrow batch: " +
                          expected);
      }
      new_chunk->arrow_array.children[i] = old_children[it->second];
    }
  }

  if (local_state.global_state) {
    local_state.global_state->record_batches.fetch_add(1);
    auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
    local_state.global_state->record_batch_rows.fetch_add(rows);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static void LanceFillDeferredColumns(ClientContext &context,
                                     const LanceScanBindData &bind_data,
                                     LanceScanGlobalState &global_state,
                                     LanceScanLocalState &local_state,
                                     DataChunk &target, idx_t pre_filter_size) {
  auto surviving_count = target.size();
  if (surviving_count == 0) {
    return;
  }

  auto rowid_idx = global_state.rowid_scan_converted_idx;
  auto &rowid_vec = local_state.scan_converted.data[rowid_idx];
  auto *rowid_data = FlatVector::GetData<uint64_t>(rowid_vec);

  vector<uint64_t> row_ids(surviving_count);
  bool filter_sliced =
      !local_state.filter_pushed_down && surviving_count < pre_filter_size;
  for (idx_t i = 0; i < surviving_count; i++) {
    row_ids[i] = filter_sliced ? rowid_data[local_state.filter_sel.get_index(i)]
                               : rowid_data[i];
  }

  vector<const char *> col_ptrs;
  col_ptrs.reserve(global_state.deferred_column_names.size());
  for (auto &name : global_state.deferred_column_names) {
    col_ptrs.push_back(name.c_str());
  }
  auto *dataset = LanceScanDatasetHandle(bind_data, global_state);
  auto take_stream = lance_create_dataset_take_stream_unfiltered(
      dataset, row_ids.data(), row_ids.size(), col_ptrs.data(),
      col_ptrs.size());
  if (!take_stream) {
    throw IOException("Failed to create deferred take stream" +
                      LanceFormatErrorSuffix());
  }

  std::unique_ptr<void, decltype(&lance_close_stream)> stream_owner(
      take_stream, lance_close_stream);

  void *batch = nullptr;
  auto rc = lance_stream_next(take_stream, &batch);
  if (rc != 0 || !batch) {
    throw IOException(rc == 1 ? "Deferred take returned no data for " +
                                    to_string(surviving_count) + " row IDs"
                              : "Failed to read deferred take batch" +
                                    LanceFormatErrorSuffix());
  }

  LanceArrowBatch exported(batch, "Failed to export deferred take batch");
  auto &take_arrow = exported.array;
  auto &take_schema = exported.schema.arrow_schema;

  // Verify take stream is exhausted (take_rows should return exactly one
  // batch).
  void *extra_batch = nullptr;
  auto extra_rc = lance_stream_next(take_stream, &extra_batch);
  if (extra_rc == 0 && extra_batch) {
    lance_free_batch(extra_batch);
    throw InternalException(
        "Deferred take returned multiple batches (expected one)");
  }
  stream_owner.reset();

  LanceCoerceArrowSchemaForDuckDB(&take_schema);

  ArrowTableSchema take_table;
  ArrowTableFunction::PopulateArrowTableSchema(context, take_table,
                                               take_schema);

  // Temporarily swap local_state fields for ArrowToDuckDB.
  // Use a scope guard to guarantee restoration even if ArrowToDuckDB throws.
  auto saved_chunk = local_state.chunk;
  auto saved_offset = local_state.chunk_offset;
  auto saved_column_ids = local_state.column_ids;

  auto restore_state = [&]() {
    local_state.chunk = saved_chunk;
    local_state.chunk_offset = saved_offset;
    local_state.column_ids = saved_column_ids;
  };

  local_state.chunk = std::move(take_arrow);
  local_state.chunk_offset = 0;
  local_state.column_ids.clear();
  for (idx_t i = 0; i < global_state.deferred_column_names.size(); i++) {
    local_state.column_ids.push_back(NumericCast<column_t>(i));
  }

  try {
    local_state.deferred_converted.Reset();
    local_state.deferred_converted.SetChildCardinality(surviving_count);
    ArrowTableFunction::ArrowToDuckDB(local_state, take_table.GetColumns(),
                                      local_state.deferred_converted);
  } catch (...) {
    restore_state();
    throw;
  }

  // Keep take Arrow batch alive: deferred_converted vectors (BLOB, LIST, etc.)
  // hold pointers into the Arrow buffers.
  local_state.deferred_arrow_owner = local_state.chunk;
  restore_state();

  for (idx_t i = 0; i < target.ColumnCount(); i++) {
    auto deferred_idx = global_state.output_to_deferred_idx[i];
    if (deferred_idx == DConstants::INVALID_INDEX) {
      continue;
    }
    if (deferred_idx >= local_state.deferred_converted.ColumnCount()) {
      throw InternalException("Deferred column index out of range");
    }
    target.data[i].Reference(local_state.deferred_converted.data[deferred_idx]);
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
  auto &arrow_columns =
      global_state.native_dataset
          ? (global_state.scan_includes_virtual_rowid
                 ? global_state.native_scan_arrow_table.GetColumns()
                 : global_state.native_arrow_table.GetColumns())
          : (global_state.scan_includes_virtual_rowid
                 ? bind_data.scan_arrow_table.GetColumns()
                 : bind_data.arrow_table.GetColumns());

  if (global_state.count_only) {
    auto start = global_state.count_only_offset.fetch_add(STANDARD_VECTOR_SIZE);
    if (start >= global_state.count_only_total_rows) {
      return;
    }
    auto output_size = MinValue<idx_t>(
        STANDARD_VECTOR_SIZE, global_state.count_only_total_rows - start);
    for (auto &column : output.data) {
      column.Reference(Value::BOOLEAN(true), count_t(output_size));
    }
    output.CheckCardinality(output_size);
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
        if (global_state.use_dataset_scanner) {
          return;
        }
        // Stream finished, try next fragment.
        local_state.fragment_pos = global_state.next_fragment_idx.fetch_add(1);
        continue;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    global_state.lines_read.fetch_add(output_size);
    local_state.rows_scanned += output_size;

    local_state.scan_converted.Reset();
    local_state.scan_converted.SetChildCardinality(output_size);
    ArrowTableFunction::ArrowToDuckDB(local_state, arrow_columns,
                                      local_state.scan_converted);

    auto fill_output_from_converted = [&](DataChunk &target) {
      if (target.ColumnCount() !=
          global_state.output_to_scan_converted_idx.size()) {
        throw InternalException(
            "Lance scan output column count does not match mapping");
      }
      for (idx_t i = 0; i < target.ColumnCount(); i++) {
        if (global_state.output_is_empty[i]) {
          target.data[i].Reference(Value::BOOLEAN(true), count_t(output_size));
          continue;
        }
        if (global_state.output_is_forced_null[i]) {
          target.data[i].Reference(Value(target.data[i].GetType()),
                                   count_t(output_size));
          continue;
        }
        auto scan_idx = global_state.output_to_scan_converted_idx[i];
        if (scan_idx == DConstants::INVALID_INDEX) {
          if (global_state.deferred_materialization &&
              i < global_state.output_to_deferred_idx.size() &&
              global_state.output_to_deferred_idx[i] !=
                  DConstants::INVALID_INDEX) {
            // Deferred column: init to constant NULL so Slice in
            // ApplyDuckDBFilters operates on a valid vector.
            target.data[i].Reference(Value(target.data[i].GetType()),
                                     count_t(output_size));
            continue;
          }
          throw InternalException(
              "Lance scan output references invalid column");
        }
        if (scan_idx >= local_state.scan_converted.ColumnCount()) {
          throw InternalException(
              "Lance scan output references invalid column");
        }
        if (global_state.output_is_duckdb_rowid[i]) {
          CastRowIdToDuckDBRowType(local_state.scan_converted.data[scan_idx],
                                   target.data[i], output_size);
          continue;
        }
        target.data[i].Reference(local_state.scan_converted.data[scan_idx]);
      }
      target.CheckCardinality(output_size);
    };

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      fill_output_from_converted(local_state.all_columns);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      if (global_state.deferred_materialization &&
          local_state.all_columns.size() > 0) {
        LanceFillDeferredColumns(context, bind_data, global_state, local_state,
                                 local_state.all_columns, output_size);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
    } else {
      fill_output_from_converted(output);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters, output,
                           local_state.filter_sel);
      }
      if (global_state.deferred_materialization && output.size() > 0) {
        LanceFillDeferredColumns(context, bind_data, global_state, local_state,
                                 output, output_size);
      }
    }

    if (output.size() == 0) {
      // Filtering can leave dictionary vectors; restore the output cache
      // before materializing the next batch.
      output.Reset();
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
  result["Lance Scan Backend"] =
      bind_data.UsesNamespaceQuery() ? "namespace_query_table" : "dataset";
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  result["Lance Sampling Pushdown"] =
      bind_data.sampling_pushed_down ? "true" : "false";
  if (bind_data.sampling_pushed_down) {
    result["Lance Sample Percentage"] = to_string(bind_data.sample_percentage);
    result["Lance Sample Seed"] = to_string(bind_data.sample_seed);
    result["Lance Sample Repeatable"] =
        bind_data.sample_repeatable ? "true" : "false";
  }
  result["Lance Limit Offset Pushdown"] =
      bind_data.limit_offset_pushed_down ? "true" : "false";
  result["Lance Limit"] = bind_data.pushed_limit.IsValid()
                              ? to_string(bind_data.pushed_limit.GetIndex())
                              : "none";
  result["Lance Offset"] = to_string(bind_data.pushed_offset);

  string filter_ir_msg;
  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    TryEncodeLanceFilterIRMessage(bind_data.lance_pushed_filter_ir_parts,
                                  filter_ir_msg);
  }

  result["Lance Filter IR Bytes (Bind)"] = to_string(filter_ir_msg.size());

  string plan;
  string error;
  if (!bind_data.UsesNamespaceQuery()) {
    if (TryLanceExplainDatasetScan(
            bind_data.dataset, nullptr,
            filter_ir_msg.empty() ? nullptr : &filter_ir_msg,
            bind_data.pushed_limit, bind_data.pushed_offset,
            bind_data.explain_verbose, plan, error)) {
      result["Lance Plan (Bind)"] = plan;
    } else if (!error.empty()) {
      result["Lance Plan Error (Bind)"] = error;
    }
  }

  return result;
}

static void LanceScanGetMetrics(TableFunctionGetMetricsInput &input) {
  input.operator_metrics.rows_scanned =
      input.local_state->Cast<LanceScanLocalState>().rows_scanned;
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
  auto &global_state = input.global_state->Cast<LanceScanGlobalState>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";
  result["Lance Dataset Cache Hit"] =
      (global_state.native_dataset ? global_state.native_dataset_cache_hit
                                   : bind_data.dataset_cache_hit)
          ? "true"
          : "false";
  result["Lance Scan Backend"] =
      global_state.use_namespace_query ? "namespace_query_table" : "dataset";
  result["Lance Scan Mode"] = global_state.use_namespace_query
                                  ? "namespace_query_table"
                              : global_state.use_dataset_scanner ? "dataset"
                                                                 : "fragment";
  result["Lance Deferred Materialization"] =
      global_state.deferred_materialization ? "true" : "false";
  if (global_state.deferred_materialization) {
    result["Lance Deferred Columns"] =
        StringUtil::Join(global_state.deferred_column_names, ", ");
  }
  result["Lance Sampling Pushdown"] =
      global_state.sampling_pushed_down ? "true" : "false";
  if (global_state.sampling_pushed_down) {
    result["Lance Sample Percentage"] =
        to_string(global_state.sample_percentage);
    result["Lance Sample Seed"] = to_string(global_state.sample_seed);
    result["Lance Sample Repeatable"] =
        global_state.sample_repeatable ? "true" : "false";
  }
  result["Lance Limit Offset Pushdown"] =
      global_state.limit_offset_pushed_down ? "true" : "false";
  result["Lance Limit"] = global_state.pushed_limit.IsValid()
                              ? to_string(global_state.pushed_limit.GetIndex())
                              : "none";
  result["Lance Offset"] = to_string(global_state.pushed_offset);
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
    input.operator_metrics.SetExtraInfo(std::move(result));
  }

  result["Lance Filter IR Bytes"] =
      to_string(global_state.lance_filter_ir.size());
  if (!global_state.scan_column_names.empty()) {
    result["Lance Projection"] =
        StringUtil::Join(global_state.scan_column_names, "\n");
  }

  if (global_state.use_namespace_query) {
    input.operator_metrics.SetExtraInfo(std::move(result));
  }

  if (!global_state.explain_computed.load()) {
    std::lock_guard<std::mutex> guard(global_state.explain_mutex);
    if (!global_state.explain_computed.load()) {
      string plan;
      string error;
      auto ok = TryLanceExplainDatasetScan(
          LanceScanDatasetHandle(bind_data, global_state),
          &global_state.scan_column_names,
          global_state.lance_filter_ir.empty() ? nullptr
                                               : &global_state.lance_filter_ir,
          global_state.pushed_limit, global_state.pushed_offset,
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

  input.operator_metrics.SetExtraInfo(std::move(result));
}

static bool TryParseConstantLimitOffset(const LogicalLimit &limit_op,
                                        optional_idx &out_limit,
                                        idx_t &out_offset) {
  auto limit_type = limit_op.limit_val.Type();
  switch (limit_type) {
  case LimitNodeType::UNSET:
    out_limit.SetInvalid();
    break;
  case LimitNodeType::CONSTANT_VALUE:
    out_limit = optional_idx(limit_op.limit_val.GetConstantValue());
    break;
  default:
    return false;
  }

  auto offset_type = limit_op.offset_val.Type();
  switch (offset_type) {
  case LimitNodeType::UNSET:
    out_offset = 0;
    break;
  case LimitNodeType::CONSTANT_VALUE:
    out_offset = limit_op.offset_val.GetConstantValue();
    break;
  default:
    return false;
  }
  return true;
}

static bool IsLanceScanTableFunction(const BoundTableFunction &fn) {
  return fn.GetName() == "__lance_scan" ||
         fn.GetName() == "__lance_table_scan" ||
         fn.GetName() == "__lance_namespace_scan";
}

static bool IsLanceRowIdColumn(const LogicalGet &get,
                               const LanceScanBindData &scan_bind,
                               const BoundColumnRefExpression &colref) {
  if (colref.Binding().table_index != get.table_index) {
    return false;
  }
  auto &col_ids = get.GetColumnIds();
  if (colref.Binding().column_index >= col_ids.size()) {
    return false;
  }

  auto col_id = col_ids[colref.Binding().column_index].GetPrimaryIndex();
  (void)scan_bind;
  return col_id == COLUMN_IDENTIFIER_ROW_ID ||
         IsLanceVirtualRowIdColumnId(col_id);
}

static bool IsLanceRowIdColumn(const LogicalGet &get,
                               const LanceScanBindData &scan_bind,
                               const BoundReferenceExpression &ref) {
  auto &col_ids = get.GetColumnIds();
  auto idx = NumericCast<idx_t>(ref.Index());
  if (idx >= col_ids.size()) {
    return false;
  }

  auto col_id = col_ids[idx].GetPrimaryIndex();
  (void)scan_bind;
  return col_id == COLUMN_IDENTIFIER_ROW_ID ||
         IsLanceVirtualRowIdColumnId(col_id);
}

static bool TryExtractTakeRowIdsFromExpression(
    const LogicalGet &get, const LanceScanBindData &scan_bind,
    const Expression &expr, vector<uint64_t> &out_row_ids) {
  auto is_row_id = [&](const Expression &candidate) {
    switch (candidate.GetExpressionClass()) {
    case ExpressionClass::BOUND_COLUMN_REF:
      return IsLanceRowIdColumn(get, scan_bind,
                                candidate.Cast<BoundColumnRefExpression>());
    case ExpressionClass::BOUND_REF:
      return IsLanceRowIdColumn(get, scan_bind,
                                candidate.Cast<BoundReferenceExpression>());
    default:
      return false;
    }
  };

  auto parse_row_id = [&](const Value &v, uint64_t &out) {
    if (!TryParseRowIdValue(v, out)) {
      throw InvalidInputException("Lance point lookup requires non-negative "
                                  "integer rowid/_rowid values");
    }
    return true;
  };

  if (expr.GetExpressionType() == ExpressionType::COMPARE_IN &&
      expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
    auto &op = expr.Cast<BoundOperatorExpression>();
    if (op.GetChildren().size() <= 1 || !op.GetChildren()[0] ||
        !is_row_id(*op.GetChildren()[0])) {
      return false;
    }

    out_row_ids.clear();
    out_row_ids.reserve(op.GetChildren().size() - 1);
    for (idx_t i = 1; i < op.GetChildren().size(); i++) {
      auto &child = op.GetChildren()[i];
      Value value;
      if (!child || !TryGetRowIdConstant(*child, value)) {
        return false;
      }
      uint64_t row_id = 0;
      parse_row_id(value, row_id);
      out_row_ids.push_back(row_id);
    }
    return true;
  }

  auto try_extract_equal = [&](const Expression &candidate, uint64_t &out) {
    if (candidate.GetExpressionType() != ExpressionType::COMPARE_EQUAL ||
        !BoundComparisonExpression::IsComparison(candidate)) {
      return false;
    }
    auto &cmp = candidate.Cast<BoundFunctionExpression>();
    if (!cmp.GetChildren()[0] || !cmp.GetChildren()[1]) {
      return false;
    }

    Value value;
    if (is_row_id(*cmp.GetChildren()[0]) &&
        TryGetRowIdConstant(*cmp.GetChildren()[1], value)) {
      parse_row_id(value, out);
      return true;
    }
    if (is_row_id(*cmp.GetChildren()[1]) &&
        TryGetRowIdConstant(*cmp.GetChildren()[0], value)) {
      parse_row_id(value, out);
      return true;
    }
    return false;
  };

  if (expr.GetExpressionType() == ExpressionType::CONJUNCTION_OR &&
      expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
    out_row_ids.clear();
    vector<const Expression *> stack;
    stack.push_back(&expr);
    while (!stack.empty()) {
      auto *current = stack.back();
      stack.pop_back();
      if (!current) {
        continue;
      }
      if (current->GetExpressionType() == ExpressionType::CONJUNCTION_OR &&
          current->GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
        auto &conj = current->Cast<BoundConjunctionExpression>();
        // Preserve left-to-right evaluation order.
        for (auto it = conj.GetChildren().rbegin();
             it != conj.GetChildren().rend(); ++it) {
          stack.push_back(it->get());
        }
        continue;
      }
      uint64_t row_id = 0;
      if (!try_extract_equal(*current, row_id)) {
        return false;
      }
      out_row_ids.push_back(row_id);
    }
    return !out_row_ids.empty();
  }

  return false;
}

static unique_ptr<LogicalOperator>
LanceRowIdInRewrite(unique_ptr<LogicalOperator> op) {
  for (auto &child : op->children) {
    child = LanceRowIdInRewrite(std::move(child));
  }

  if (op->type == LogicalOperatorType::LOGICAL_GET) {
    auto &get = op->Cast<LogicalGet>();
    if (!IsLanceScanTableFunction(get.function) || !get.bind_data) {
      return op;
    }
    auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();
    if (scan_bind.UsesNamespaceQuery()) {
      return op;
    }

    const auto &col_ids = get.GetColumnIds();
    ProjectionIndex filter_index;
    column_t filter_col_id = COLUMN_IDENTIFIER_ROW_ID;
    optional_ptr<const TableFilter> rowid_filter;
    for (idx_t i = 0; i < col_ids.size(); i++) {
      auto id = col_ids[i].GetPrimaryIndex();
      if (id == LANCE_COLUMN_IDENTIFIER_ROW_ID ||
          id == COLUMN_IDENTIFIER_ROW_ID) {
        auto candidate =
            get.table_filters.TryGetFilterByColumnIndex(ProjectionIndex(i));
        if (candidate) {
          filter_index = ProjectionIndex(i);
          filter_col_id = id;
          rowid_filter = candidate;
          break;
        }
      }
    }
    if (!rowid_filter) {
      return op;
    }

    auto col_type = filter_col_id == COLUMN_IDENTIFIER_ROW_ID
                        ? LogicalType::ROW_TYPE
                        : LogicalType::UBIGINT;
    auto colref = make_uniq<BoundColumnRefExpression>(
        col_type, ColumnBinding(get.table_index, filter_index));
    vector<uint64_t> row_ids;
    bool can_take = scan_bind.take_row_ids.empty();
    if (can_take) {
      auto filter_expr = rowid_filter->ToExpression(*colref);
      can_take = TryExtractTakeRowIdsFromExpression(*filter_expr, row_ids) &&
                 !row_ids.empty();
    }
    if (can_take) {
      scan_bind.take_row_ids = std::move(row_ids);
      get.table_filters.RemoveFilterByColumnIndex(filter_index);
      return op;
    }
    auto filter_expr = rowid_filter->ToExpression(*colref);
    get.table_filters.RemoveFilterByColumnIndex(filter_index);

    // DuckDB may have pruned the rowid column from the scan projection after
    // moving the predicate into table filters (only the referenced data
    // columns remain in `projection_ids`). The recreated LogicalFilter above
    // references the rowid binding, so the scan must keep producing it.
    bool rowid_projected = false;
    for (auto proj_id : get.projection_ids) {
      if (proj_id == filter_index) {
        rowid_projected = true;
        break;
      }
    }
    if (!rowid_projected) {
      get.projection_ids.push_back(filter_index);
    }

    auto estimated = op->estimated_cardinality;
    auto filter = make_uniq<LogicalFilter>();
    filter->expressions.push_back(std::move(filter_expr));
    filter->children.push_back(std::move(op));
    filter->estimated_cardinality = estimated;
    return std::move(filter);
  }

  if (op->type != LogicalOperatorType::LOGICAL_FILTER ||
      op->children.size() != 1 || !op->children[0]) {
    return op;
  }

  auto &filter_op = op->Cast<LogicalFilter>();
  auto *node = op->children[0].get();
  while (node && node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
    if (node->children.empty() || !node->children[0]) {
      return op;
    }
    node = node->children[0].get();
  }
  if (!node || node->type != LogicalOperatorType::LOGICAL_GET) {
    return op;
  }

  auto &get = node->Cast<LogicalGet>();
  if (!IsLanceScanTableFunction(get.function) || !get.bind_data) {
    return op;
  }
  auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();
  if (scan_bind.UsesNamespaceQuery()) {
    return op;
  }

  vector<uint64_t> row_ids;
  idx_t idx = 0;
  bool found = false;
  for (idx_t i = 0; i < filter_op.expressions.size(); i++) {
    if (TryExtractTakeRowIdsFromExpression(
            get, scan_bind, *filter_op.expressions[i], row_ids)) {
      found = true;
      idx = i;
      break;
    }
  }
  if (!found) {
    return op;
  }

  scan_bind.take_row_ids = row_ids;
  filter_op.expressions.erase(filter_op.expressions.begin() +
                              NumericCast<std::ptrdiff_t>(idx));
  if (filter_op.expressions.empty()) {
    auto child = std::move(op->children[0]);
    child->estimated_cardinality = op->estimated_cardinality;
    return child;
  }
  return op;
}

static bool TryBuildLanceFilterIRThroughProjections(
    const LogicalFilter &filter, const LogicalGet &get,
    const vector<string> &names, const vector<LogicalType> &types,
    const vector<string> &json_columns, const Expression &expression,
    string &out_ir) {
  auto rewritten = expression.Copy();
  auto *node = filter.children[0].get();

  while (node && node != &get) {
    if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      auto &projection = node->Cast<LogicalProjection>();
      bool valid = true;
      ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
          rewritten, [&](BoundColumnRefExpression &column_ref,
                         unique_ptr<Expression> &current) {
            if (column_ref.Depth() != 0 ||
                column_ref.Binding().table_index != projection.table_index ||
                column_ref.Binding().column_index >=
                    projection.expressions.size()) {
              valid = false;
              return;
            }

            auto replacement =
                projection.expressions[column_ref.Binding().column_index]
                    ->Copy();
            if (!column_ref.GetAlias().empty()) {
              replacement->SetAlias(column_ref.GetAlias());
            }
            current = std::move(replacement);
          });
      if (!valid) {
        return false;
      }
    } else if (node->type != LogicalOperatorType::LOGICAL_FILTER) {
      return false;
    }

    if (node->children.size() != 1 || !node->children[0]) {
      return false;
    }
    node = node->children[0].get();
  }

  return node == &get &&
         TryBuildLanceExprFilterIR(get, names, types, json_columns, false,
                                   *rewritten, out_ir);
}

static unique_ptr<LogicalOperator>
LanceLimitOffsetPushdown(unique_ptr<LogicalOperator> op) {
  for (auto &child : op->children) {
    child = LanceLimitOffsetPushdown(std::move(child));
  }

  if (op->type == LogicalOperatorType::LOGICAL_GET) {
    auto &get = op->Cast<LogicalGet>();
    if (!IsLanceScanTableFunction(get.function) || !get.bind_data) {
      return op;
    }

    auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();
    scan_bind.sampling_pushed_down = false;
    scan_bind.sample_percentage = 0.0;
    scan_bind.sample_seed = -1;
    scan_bind.sample_repeatable = false;

    if (get.extra_info.sample_options &&
        get.extra_info.sample_options->method == SampleMethod::SYSTEM_SAMPLE &&
        get.extra_info.sample_options->is_percentage) {
      scan_bind.sampling_pushed_down = true;
      scan_bind.sample_percentage = get.extra_info.sample_options->sample_size
                                        .DefaultCastAs(LogicalType::DOUBLE)
                                        .GetValue<double>();
      scan_bind.sample_seed = get.extra_info.sample_options->GetSeed();
      scan_bind.sample_repeatable = get.extra_info.sample_options->repeatable;
    }
    return op;
  }

  if (op->type != LogicalOperatorType::LOGICAL_LIMIT) {
    return op;
  }

  auto &limit_op = op->Cast<LogicalLimit>();
  optional_idx pushed_limit = optional_idx::Invalid();
  idx_t pushed_offset = 0;
  if (!TryParseConstantLimitOffset(limit_op, pushed_limit, pushed_offset)) {
    return op;
  }
  if (op->children.empty() || !op->children[0]) {
    return op;
  }

  auto *node = op->children[0].get();
  vector<LogicalFilter *> intermediate_filters;
  while (node && (node->type == LogicalOperatorType::LOGICAL_PROJECTION ||
                  node->type == LogicalOperatorType::LOGICAL_FILTER)) {
    if (node->type == LogicalOperatorType::LOGICAL_FILTER) {
      intermediate_filters.push_back(&node->Cast<LogicalFilter>());
    }
    if (node->children.empty() || !node->children[0]) {
      return op;
    }
    node = node->children[0].get();
  }
  if (!node || node->type != LogicalOperatorType::LOGICAL_GET) {
    return op;
  }

  auto &get = node->Cast<LogicalGet>();
  if (!IsLanceScanTableFunction(get.function) || !get.bind_data) {
    return op;
  }
  if (get.extra_info.sample_options) {
    // Sampling must occur before limiting, so do not remove the LIMIT node.
    return op;
  }

  // If we traversed through FILTER nodes, verify ALL filter expressions can be
  // handled by Lance. pushdown_complex_filter already pushed these into
  // lance_pushed_filter_ir_parts, so Lance will execute filter-then-limit
  // internally. The LOGICAL_FILTER stays in the plan as a safety fallback.
  // If any expression is not pushable, Lance would apply limit before the
  // residual DuckDB filter, producing wrong results — so bail out.
  if (!intermediate_filters.empty()) {
    auto &scan_bind_check = get.bind_data->Cast<LanceScanBindData>();
    for (auto *filter_node : intermediate_filters) {
      for (auto &expr : filter_node->expressions) {
        if (!expr) {
          continue;
        }
        string discard_ir;
        if (!TryBuildLanceFilterIRThroughProjections(
                *filter_node, get, scan_bind_check.names, scan_bind_check.types,
                scan_bind_check.json_columns, *expr, discard_ir)) {
          return op;
        }
      }
    }
  }

  auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();
  if (scan_bind.UsesNamespaceQuery()) {
    auto max_query_value =
        static_cast<idx_t>(std::numeric_limits<int32_t>::max());
    if ((!pushed_limit.IsValid() && pushed_offset != 0) ||
        (pushed_limit.IsValid() && pushed_limit.GetIndex() > max_query_value) ||
        pushed_offset > max_query_value) {
      return op;
    }
  }
  if (!scan_bind.take_row_ids.empty()) {
    return op;
  }
  auto &col_ids = get.GetColumnIds();
  for (auto &entry : get.table_filters) {
    if (entry.GetIndex() >= col_ids.size()) {
      continue;
    }
    auto col_id = col_ids[entry.GetIndex()].GetPrimaryIndex();
    if (col_id != COLUMN_IDENTIFIER_ROW_ID &&
        !IsLanceVirtualRowIdColumnId(col_id)) {
      continue;
    }
    if (IsTakeRowIdFilter(entry.Filter())) {
      return op;
    }
  }

  // Pre-check filter pushability: limit/offset pushdown requires filter
  // pushdown (otherwise Lance would apply LIMIT before DuckDB's filter,
  // producing wrong results). If any table filter cannot be encoded to
  // Lance IR (e.g., LIKE with interior wildcards, unsupported column types,
  // or 'infinity' literals), keep the LogicalLimit in the plan and skip
  // limit pushdown. DuckDB then applies LIMIT on top of the filtered
  // stream — slower than Lance-side pushdown, but correct and strictly
  // better than the previous "IO Error: Lance limit/offset pushdown
  // requires filter pushdown" failure.
  auto probe = ProbeLanceTableFilterIR(get, scan_bind.names, scan_bind.types,
                                       scan_bind.json_columns);
  if (!probe.all_filters_pushed) {
    return op;
  }

  scan_bind.limit_offset_pushed_down = true;
  scan_bind.pushed_limit = pushed_limit;
  scan_bind.pushed_offset = pushed_offset;

  auto child = std::move(op->children[0]);
  child->estimated_cardinality = op->estimated_cardinality;
  return child;
}

static void
LanceLimitOffsetPushdownOptimizer(OptimizerExtensionInput &,
                                  unique_ptr<LogicalOperator> &plan) {
  plan = LanceLimitOffsetPushdown(std::move(plan));
}

// The early pass runs before constant folding. Fold only immutable constants
// in a private copy, retaining the original predicate when pushdown is
// rejected.
static void FoldLanceExecFilterConstants(ClientContext &context,
                                         unique_ptr<Expression> &expr) {
  ExpressionIterator::EnumerateChildren(
      *expr, [&](unique_ptr<Expression> &child) {
        FoldLanceExecFilterConstants(context, child);
      });
  if (!expr->HasParameter() && !expr->IsVolatile() && expr->IsFoldable()) {
    Value value;
    if (ExpressionExecutor::TryEvaluateScalar(context, *expr, value)) {
      expr = make_uniq<BoundConstantExpression>(std::move(value));
    }
  }
}

// Resolve view/projection bindings before encoding a filter in the early pass.
// Each projection is visited from the filter down towards the scan.
static void InlineLanceFilterProjection(unique_ptr<Expression> &expr,
                                        const LogicalProjection &projection) {
  if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
    const auto &ref = expr->Cast<BoundColumnRefExpression>();
    if (ref.Depth() == 0 &&
        ref.Binding().table_index == projection.table_index) {
      expr = projection.GetExpression(ref.Binding()).Copy();
      return;
    }
  }
  ExpressionIterator::EnumerateChildren(
      *expr, [&](unique_ptr<Expression> &child) {
        InlineLanceFilterProjection(child, projection);
      });
}

static unique_ptr<LogicalOperator>
LanceExecPushdown(ClientContext &context, Optimizer &optimizer,
                  unique_ptr<LogicalOperator> op, bool preserve_avg = false) {
  auto failfast = []() {
    return std::getenv("LANCE_EXEC_PUSHDOWN_FAILFAST") != nullptr;
  };

  auto try_rewrite =
      [&](LogicalAggregate &agg_op, const LogicalOrder *order_op,
          const LogicalProjection *post_projection,
          const vector<LogicalType> &expected_types,
          idx_t estimated_cardinality) -> unique_ptr<LogicalOperator> {
    auto bail = [&](const string &msg) -> unique_ptr<LogicalOperator> {
      if (failfast()) {
        throw IOException(msg);
      }
      return nullptr;
    };

    if (preserve_avg) {
      bool has_avg = false;
      for (const auto &expr : agg_op.expressions) {
        if (expr->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE &&
            expr->Cast<BoundAggregateExpression>().Function().GetName() ==
                "avg") {
          has_avg = true;
          break;
        }
      }
      if (!has_avg) {
        return nullptr;
      }
    }

    if (agg_op.children.size() != 1 || !agg_op.children[0]) {
      return bail("Lance exec pushdown: aggregate child shape not supported");
    }

    vector<const vector<unique_ptr<Expression>> *> projection_stack;
    projection_stack.reserve(2);
    vector<unique_ptr<Expression>> *filter_exprs = nullptr;
    vector<const LogicalProjection *> filter_projections;

    auto *child = agg_op.children[0].get();
    while (child) {
      if (child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        auto &proj = child->Cast<LogicalProjection>();
        projection_stack.push_back(&proj.expressions);
        if (filter_exprs) {
          filter_projections.push_back(&proj);
        }
        if (child->children.size() != 1 || !child->children[0]) {
          return bail(
              "Lance exec pushdown: projection child shape not supported");
        }
        child = child->children[0].get();
        continue;
      }
      if (child->type == LogicalOperatorType::LOGICAL_FILTER) {
        if (filter_exprs) {
          return bail("Lance exec pushdown: multiple filters not supported");
        }
        auto &filter_op = child->Cast<LogicalFilter>();
        filter_exprs = &filter_op.expressions;
        if (child->children.size() != 1 || !child->children[0]) {
          return bail("Lance exec pushdown: filter child shape not supported");
        }
        child = child->children[0].get();
        continue;
      }
      break;
    }

    if (!child || child->type != LogicalOperatorType::LOGICAL_GET) {
      return bail("Lance exec pushdown: expected LogicalGet child");
    }
    auto &scan_get = child->Cast<LogicalGet>();
    if (!IsLanceScanTableFunction(scan_get.function) || !scan_get.bind_data) {
      return bail("Lance exec pushdown: expected Lance scan LogicalGet");
    }
    auto &scan_bind = scan_get.bind_data->Cast<LanceScanBindData>();
    if (scan_bind.UsesNamespaceQuery()) {
      return bail(
          "Lance exec pushdown: namespace query_table scans not supported");
    }

    vector<string> filter_parts;
    string filter_ir_msg;
    vector<idx_t> extra_scan_col_ids;
    extra_scan_col_ids.reserve(scan_get.table_filters.FilterCount());

    if (scan_get.table_filters.HasFilters() ||
        scan_get.table_filters.HasMultiColumnFilters()) {
      // rowid filters have exec-pushdown-specific semantics; the shared
      // probe below only covers IR encodability, so bail early here.
      for (auto &it : scan_get.table_filters) {
        auto col_id = scan_get.GetColumnIds()[it.GetIndex()].GetPrimaryIndex();
        if (col_id >= scan_bind.names.size() ||
            col_id >= scan_bind.types.size()) {
          return bail(
              "Lance exec pushdown: table_filters column out of bounds");
        }
        if (col_id == COLUMN_IDENTIFIER_ROW_ID ||
            IsLanceVirtualRowIdColumnId(col_id)) {
          return bail("Lance exec pushdown: rowid filters not supported");
        }
      }

      auto table_filters = ProbeLanceTableFilterIR(
          scan_get, scan_bind.names, scan_bind.types, scan_bind.json_columns);
      if (!table_filters.all_filters_pushed) {
        return bail("Lance exec pushdown: table_filters not fully pushable");
      }
      filter_parts = std::move(table_filters.parts);

      for (auto &it : scan_get.table_filters) {
        auto col_id = scan_get.GetColumnIds()[it.GetIndex()].GetPrimaryIndex();
        extra_scan_col_ids.push_back(col_id);
      }
    }

    for (const auto &filter : scan_get.table_filters.GetMultiColumnFilters()) {
      const auto &expression_filter = ExpressionFilter::GetExpressionFilter(
          *filter, "Lance aggregate pushdown");
      for (auto projection : expression_filter.column_indexes) {
        extra_scan_col_ids.push_back(
            scan_get.GetColumnIds()[projection].GetPrimaryIndex());
      }
    }

    if (filter_exprs && !filter_exprs->empty()) {
      filter_parts.reserve(filter_parts.size() + filter_exprs->size());
      for (auto &original : *filter_exprs) {
        auto expr = original ? original->Copy() : nullptr;
        if (expr) {
          for (auto projection : filter_projections) {
            InlineLanceFilterProjection(expr, *projection);
          }
          FoldLanceExecFilterConstants(context, expr);
        }
        if (!expr || expr->HasParameter() || expr->IsVolatile() ||
            expr->CanThrow()) {
          return bail("Lance exec pushdown: filter expression not pushable");
        }
        string part;
        if (!TryBuildLanceExprFilterIR(scan_get, scan_bind.names,
                                       scan_bind.types, scan_bind.json_columns,
                                       false, *expr, part)) {
          return bail("Lance exec pushdown: filter IR encode failed");
        }
        filter_parts.push_back(std::move(part));
      }
    }

    if (!scan_bind.lance_pushed_filter_ir_parts.empty()) {
      filter_parts.reserve(filter_parts.size() +
                           scan_bind.lance_pushed_filter_ir_parts.size());
      for (auto &part : scan_bind.lance_pushed_filter_ir_parts) {
        filter_parts.push_back(part);
      }
    }

    if (!filter_parts.empty()) {
      if (!TryEncodeLanceFilterIRMessage(filter_parts, filter_ir_msg)) {
        return bail("Lance exec pushdown: filter message encode failed");
      }
    }

    string exec_ir;
    if (!TryEncodeLanceExecIRv1(scan_get, scan_bind, filter_ir_msg,
                                extra_scan_col_ids, projection_stack, agg_op,
                                order_op, post_projection, exec_ir)) {
      if (failfast()) {
        throw IOException("Lance exec pushdown: ExecIR encode failed");
      }
      return nullptr;
    }

    vector<LogicalType> exec_types;
    vector<string> exec_names;
    unique_ptr<LanceExecBindData> exec_bind;
    try {
      exec_bind = make_uniq<LanceExecBindData>();
      exec_bind->file_path = scan_bind.file_path;
      exec_bind->exec_ir = exec_ir;
      exec_bind->dataset_entry = LanceGetOrOpenDatasetEntry(
          context, exec_bind->file_path, &exec_bind->dataset_cache_hit);
      exec_bind->dataset = exec_bind->dataset_entry
                               ? exec_bind->dataset_entry->Handle()
                               : nullptr;
      if (!exec_bind->dataset) {
        throw IOException("Failed to open Lance dataset for exec pushdown: " +
                          exec_bind->file_path + LanceFormatErrorSuffix());
      }

      auto *schema_handle = lance_get_exec_schema(
          exec_bind->dataset,
          exec_bind->exec_ir.empty()
              ? nullptr
              : reinterpret_cast<const uint8_t *>(exec_bind->exec_ir.data()),
          exec_bind->exec_ir.size());
      if (!schema_handle) {
        throw IOException("Failed to validate Lance exec IR: " +
                          LanceConsumeLastError());
      }
      memset(&exec_bind->schema_root.arrow_schema, 0,
             sizeof(exec_bind->schema_root.arrow_schema));
      if (lance_schema_to_arrow(schema_handle,
                                &exec_bind->schema_root.arrow_schema) != 0) {
        lance_free_schema(schema_handle);
        throw IOException(
            "Failed to export exec schema to Arrow C interface: " +
            LanceFormatErrorSuffix());
      }
      lance_free_schema(schema_handle);
      LanceCoerceArrowSchemaForDuckDB(&exec_bind->schema_root.arrow_schema);
      ArrowTableFunction::PopulateArrowTableSchema(
          context, exec_bind->arrow_table, exec_bind->schema_root.arrow_schema);
      exec_names = exec_bind->arrow_table.GetNames();
      exec_types = exec_bind->arrow_table.GetTypes();
    } catch (...) {
      if (failfast()) {
        throw;
      }
      return nullptr;
    }

    if (exec_types.size() != expected_types.size()) {
      if (failfast()) {
        throw IOException("Lance exec pushdown: type count mismatch");
      }
      return nullptr;
    }
    for (idx_t i = 0; i < exec_types.size(); i++) {
      if (exec_types[i] != expected_types[i]) {
        if (failfast()) {
          throw IOException("Lance exec pushdown: type mismatch at column " +
                            to_string(i) + ": expected " +
                            expected_types[i].ToString() + ", got " +
                            exec_types[i].ToString());
        }
        return nullptr;
      }
    }

    // The child LogicalGet is used for physical planning/execution only. We
    // wrap it in LogicalLanceExec to preserve the original grouped aggregate
    // column bindings (group_index/aggregate_index) for upstream operators.
    auto exec_get = make_uniq<LogicalGet>(
        optimizer.binder.GenerateTableIndex(),
        BoundTableFunction(LanceExecFunction()), std::move(exec_bind),
        exec_types, StringsToIdentifiers(exec_names));
    exec_get->parameters.push_back(
        Value(exec_get->bind_data->Cast<LanceExecBindData>().file_path));
    exec_get->parameters.push_back(Value::BLOB_RAW(exec_ir));

    vector<ColumnIndex> column_ids;
    column_ids.reserve(exec_types.size());
    for (idx_t i = 0; i < exec_types.size(); i++) {
      column_ids.emplace_back(NumericCast<column_t>(i));
    }
    exec_get->SetColumnIds(std::move(column_ids));
    exec_get->SetEstimatedCardinality(estimated_cardinality);

    auto exec = make_uniq<LogicalLanceExec>(
        agg_op.group_index, agg_op.aggregate_index, agg_op.groups.size(),
        expected_types, std::move(exec_get));
    exec->SetEstimatedCardinality(estimated_cardinality);
    return std::move(exec);
  };

  if (op->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
    auto &order = op->Cast<LogicalOrder>();
    if (op->children.size() == 1 && op->children[0]) {
      auto *child = op->children[0].get();
      const LogicalProjection *post_projection = nullptr;
      if (child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        post_projection = &child->Cast<LogicalProjection>();
        if (child->children.size() != 1 || !child->children[0]) {
          return op;
        }
        child = child->children[0].get();
      }
      if (child &&
          child->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        auto &agg = child->Cast<LogicalAggregate>();
        if (auto rewritten =
                try_rewrite(agg, &order, post_projection, op->types,
                            child->estimated_cardinality)) {
          return rewritten;
        }
      }
    }
  }

  for (auto &child : op->children) {
    child =
        LanceExecPushdown(context, optimizer, std::move(child), preserve_avg);
  }

  if (op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    return op;
  }
  auto &agg = op->Cast<LogicalAggregate>();
  if (auto rewritten = try_rewrite(agg, nullptr, nullptr, op->types,
                                   op->estimated_cardinality)) {
    return rewritten;
  }
  return op;
}

static void LanceExecPushdownOptimizer(OptimizerExtensionInput &input,
                                       unique_ptr<LogicalOperator> &plan) {
  plan = LanceExecPushdown(input.context, input.optimizer, std::move(plan));
}

static void LanceRowIdInRewriteOptimizer(OptimizerExtensionInput &,
                                         unique_ptr<LogicalOperator> &plan) {
  plan = LanceRowIdInRewrite(std::move(plan));
}

static unique_ptr<LogicalOperator>
LanceLikePushdown(unique_ptr<LogicalOperator> op) {
  for (auto &child : op->children) {
    child = LanceLikePushdown(std::move(child));
  }

  if (op->type != LogicalOperatorType::LOGICAL_FILTER ||
      op->children.size() != 1 || !op->children[0]) {
    return op;
  }

  auto &filter_op = op->Cast<LogicalFilter>();
  auto *node = op->children[0].get();
  while (node && node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
    if (node->children.empty() || !node->children[0]) {
      return op;
    }
    node = node->children[0].get();
  }
  if (!node || node->type != LogicalOperatorType::LOGICAL_GET) {
    return op;
  }

  auto &get = node->Cast<LogicalGet>();
  if (!IsLanceScanTableFunction(get.function) || !get.bind_data) {
    return op;
  }
  auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();

  for (auto &expr : filter_op.expressions) {
    if (!expr || expr->HasParameter() || expr->IsVolatile() ||
        expr->CanThrow()) {
      continue;
    }
    if (expr->GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
      continue;
    }
    auto &func = expr->Cast<BoundFunctionExpression>();
    auto &name = func.Function().GetName().GetIdentifierName();
    if (name != "~~" && name != "~~*" && name != "like_escape" &&
        name != "ilike_escape") {
      continue;
    }

    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types,
                                   scan_bind.json_columns, false, *expr,
                                   filter_ir)) {
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));
  }

  return op;
}

static void LanceLikePushdownOptimizer(OptimizerExtensionInput &,
                                       unique_ptr<LogicalOperator> &plan) {
  plan = LanceLikePushdown(std::move(plan));
}

static unique_ptr<LogicalOperator>
LanceCardinalityFixup(ClientContext &context, unique_ptr<LogicalOperator> op) {
  for (auto &child : op->children) {
    child = LanceCardinalityFixup(context, std::move(child));
  }

  if (op->type != LogicalOperatorType::LOGICAL_GET) {
    return op;
  }

  auto &get = op->Cast<LogicalGet>();
  if (!IsLanceScanTableFunction(get.function) || !get.bind_data) {
    if (get.function.GetName().GetIdentifierName() != "__lance_scan" ||
        get.parameters.size() != 1 || get.parameters[0].IsNull() ||
        get.parameters[0].type() != LogicalType::VARCHAR) {
      return op;
    }

    auto path = get.parameters[0].GetValue<string>();
    auto *dataset = LanceOpenDataset(context, path);
    if (!dataset) {
      return op;
    }
    auto rows = lance_dataset_count_rows(dataset);
    lance_close_dataset(dataset);
    if (rows < 0) {
      return op;
    }
    get.SetEstimatedCardinality(NumericCast<idx_t>(rows));
    return op;
  }

  auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();
  if (!scan_bind.take_row_ids.empty()) {
    get.SetEstimatedCardinality(scan_bind.take_row_ids.size());
    return op;
  }

  if (!scan_bind.dataset) {
    return op;
  }

  auto rows = lance_dataset_count_rows(scan_bind.dataset);
  if (rows < 0) {
    return op;
  }
  get.SetEstimatedCardinality(NumericCast<idx_t>(rows));
  return op;
}

static void LanceCardinalityFixupOptimizer(OptimizerExtensionInput &input,
                                           unique_ptr<LogicalOperator> &plan) {
  plan = LanceCardinalityFixup(input.context, std::move(plan));
}

void RegisterLanceScanOptimizer(DBConfig &config) {
  OptimizerExtension exec_ext;
  // Preserve native AVG before DuckDB rewrites it into SUM/COUNT. Retain
  // the later pass for plans that first need DuckDB's filter/projection
  // rewrites.
  exec_ext.pre_optimize_function = [](OptimizerExtensionInput &input,
                                      unique_ptr<LogicalOperator> &plan) {
    plan->ResolveOperatorTypes();
    plan = LanceExecPushdown(input.context, input.optimizer, std::move(plan),
                             true);
  };
  exec_ext.optimize_function = LanceExecPushdownOptimizer;
  OptimizerExtension::Register(config, std::move(exec_ext));

  OptimizerExtension rowid_take_ext;
  rowid_take_ext.optimize_function = LanceRowIdInRewriteOptimizer;
  OptimizerExtension::Register(config, std::move(rowid_take_ext));

  OptimizerExtension like_ext;
  like_ext.optimize_function = LanceLikePushdownOptimizer;
  OptimizerExtension::Register(config, std::move(like_ext));

  OptimizerExtension limit_ext;
  limit_ext.optimize_function = LanceLimitOffsetPushdownOptimizer;
  OptimizerExtension::Register(config, std::move(limit_ext));

  OptimizerExtension cardinality_ext;
  cardinality_ext.optimize_function = LanceCardinalityFixupOptimizer;
  OptimizerExtension::Register(config, std::move(cardinality_ext));
}

// ---- __lance_exec (internal-only) ----

struct LanceExecGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};

  vector<idx_t> projection_ids;

  idx_t MaxThreads() const override { return 1; }
};

struct LanceExecLocalState : public ArrowScanLocalState {
  explicit LanceExecLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                               ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context) {}

  void *stream = nullptr;
  LanceExecGlobalState *global_state = nullptr;

  ~LanceExecLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static unique_ptr<FunctionData> LanceExecBind(ClientContext &context,
                                              TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types,
                                              vector<Identifier> &names) {
  if (input.inputs.size() != 2 || input.inputs[0].IsNull() ||
      input.inputs[1].IsNull()) {
    throw InvalidInputException("__lance_exec requires (path, exec_ir)");
  }

  auto result = make_uniq<LanceExecBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  result->exec_ir =
      input.inputs[1].DefaultCastAs(LogicalType::BLOB).GetValue<string>();

  result->dataset_entry = LanceGetOrOpenDatasetEntry(
      context, result->file_path, &result->dataset_cache_hit);
  result->dataset =
      result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_exec_schema(
      result->dataset,
      result->exec_ir.empty()
          ? nullptr
          : reinterpret_cast<const uint8_t *>(result->exec_ir.data()),
      result->exec_ir.size());
  if (!schema_handle) {
    throw IOException("Failed to validate Lance exec IR: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance exec schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&result->schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();

  names.clear();
  for (const auto &name : result->names) {
    names.emplace_back(name);
  }
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceExecInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto state = make_uniq_base<GlobalTableFunctionState, LanceExecGlobalState>();
  auto &global = state->Cast<LanceExecGlobalState>();
  global.projection_ids = input.projection_ids;
  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceExecLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                   GlobalTableFunctionState *global_state) {
  auto &bind_data = input.bind_data->Cast<LanceExecBindData>();
  auto &global = global_state->Cast<LanceExecGlobalState>();

  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result =
      make_uniq<LanceExecLocalState>(std::move(chunk), context.client);
  result->global_state = &global;
  LanceExecContext exec_ctx;
  exec_ctx.threads = NumericCast<uint32_t>(
      DBConfig::GetConfig(context.client).options.maximum_threads);
  result->stream = lance_create_dataset_exec_stream_ir(
      bind_data.dataset,
      bind_data.exec_ir.empty()
          ? nullptr
          : reinterpret_cast<const uint8_t *>(bind_data.exec_ir.data()),
      bind_data.exec_ir.size(), &exec_ctx);
  if (!result->stream) {
    throw IOException("Failed to create Lance exec stream" +
                      LanceFormatErrorSuffix());
  }
  return std::move(result);
}

static bool LanceExecLoadNextBatch(LanceExecLocalState &local_state) {
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

  LanceArrowBatch exported(batch);
  auto &new_chunk = exported.array;

  if (local_state.global_state) {
    local_state.global_state->record_batches.fetch_add(1);
    auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
    local_state.global_state->record_batch_rows.fetch_add(rows);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static void LanceExecFunc(ClientContext &context, TableFunctionInput &data,
                          DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceExecBindData>();
  auto &global_state = data.global_state->Cast<LanceExecGlobalState>();
  auto &local_state = data.local_state->Cast<LanceExecLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceExecLoadNextBatch(local_state)) {
        return;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    global_state.lines_read.fetch_add(output_size);

    output.SetChildCardinality(output_size);
    ArrowTableFunction::ArrowToDuckDB(
        local_state, bind_data.arrow_table.GetColumns(), output, false);
    local_state.chunk_offset += output_size;

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

static InsertionOrderPreservingMap<string>
LanceExecToString(TableFunctionToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceExecBindData>();
  result["Lance Path"] = bind_data.file_path;
  result["Lance Exec IR Bytes"] = to_string(bind_data.exec_ir.size());
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  return result;
}

static void LanceExecGetMetrics(TableFunctionGetMetricsInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceExecBindData>();
  auto &global_state = input.global_state->Cast<LanceExecGlobalState>();
  result["Lance Path"] = bind_data.file_path;
  result["Lance Exec IR Bytes"] = to_string(bind_data.exec_ir.size());
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());
  input.operator_metrics.SetExtraInfo(std::move(result));
}

static TableFunction LanceExecFunction() {
  TableFunction function(
      "__lance_exec", {LogicalType::VARCHAR, LogicalType::BLOB}, LanceExecFunc,
      LanceExecBind, LanceExecInitGlobal, LanceExecLocalInit);
  function.to_string = LanceExecToString;
  function.get_metrics = LanceExecGetMetrics;
  return function;
}

static TableFunction LanceTableScanFunction() {
  TableFunction function("__lance_table_scan", {}, LanceScanFunc);
  function.projection_pushdown = true;
  function.filter_pushdown = true;
  function.filter_prune = true;
  function.sampling_pushdown = true;
  function.statistics = LanceScanStatistics;
  function.cardinality = LanceScanCardinality;
  function.get_partition_stats = LanceScanGetPartitionStats;
  function.supports_pushdown_type = LanceSupportsPushdownType;
  function.pushdown_expression = LancePushdownExpression;
  function.pushdown_complex_filter = LancePushdownComplexFilter;
  function.pushdown_expression = LancePushdownExpression;
  function.get_virtual_columns = LanceGetVirtualColumns;
  function.to_string = LanceScanToString;
  function.get_metrics = LanceScanGetMetrics;
  function.get_table_entry = [](optional_ptr<const FunctionData> bind_data)
      -> optional_ptr<TableCatalogEntry> {
    const auto &scan_bind = bind_data->Cast<LanceScanBindData>();
    return scan_bind.table_entry;
  };
  function.init_global = LanceScanInitGlobal;
  function.init_local = LanceScanLocalInit;
  return function;
}

static void PopulateNamespaceQueryScanSchema(ClientContext &context,
                                             const LanceTableEntry &table,
                                             LanceScanBindData &result) {
  vector<string> field_names;
  vector<LogicalType> field_types;
  field_names.reserve(table.GetColumns().PhysicalColumnCount());
  field_types.reserve(table.GetColumns().PhysicalColumnCount());
  for (auto &column : table.GetColumns().Physical()) {
    field_names.push_back(column.Name().GetIdentifierName());
    field_types.push_back(column.Type());
  }

  memset(&result.schema_root.arrow_schema, 0,
         sizeof(result.schema_root.arrow_schema));
  auto properties = context.GetClientProperties();
  ArrowConverter::ToArrowSchema(&result.schema_root.arrow_schema, field_types,
                                field_names, properties);
  LanceCoerceArrowSchemaForDuckDB(&result.schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(context, result.arrow_table,
                                               result.schema_root.arrow_schema);
  result.names = result.arrow_table.GetNames();
  result.types = result.arrow_table.GetTypes();
  result.json_columns = table.JsonColumnNames();
  result.blob_columns = table.GetBlobColumns();

  field_names.push_back(LANCE_ROW_ID_COLUMN_NAME);
  field_types.push_back(LogicalType::UBIGINT);
  memset(&result.scan_schema_root.arrow_schema, 0,
         sizeof(result.scan_schema_root.arrow_schema));
  ArrowConverter::ToArrowSchema(&result.scan_schema_root.arrow_schema,
                                field_types, field_names, properties);
  LanceCoerceArrowSchemaForDuckDB(&result.scan_schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result.scan_arrow_table, result.scan_schema_root.arrow_schema);
}

TableFunction
LanceTableEntry::GetScanFunction(ClientContext &context,
                                 unique_ptr<FunctionData> &bind_data) {
  auto result = make_uniq<LanceScanBindData>();
  result->table_entry = this;
  result->file_path = dataset_uri;

  if (IsNamespaceBacked() && NamespaceConfig().IsRest() &&
      LanceNamespaceQueryTableEnabled(context)) {
    result->namespace_query_config =
        make_uniq<LanceNamespaceTableConfig>(NamespaceConfig());
    PopulateNamespaceQueryScanSchema(context, *this, *result);
    bind_data = std::move(result);
    auto function = LanceTableScanFunction();
    function.sampling_pushdown = false;
    return function;
  }

  string display_uri;
  result->dataset_entry = LanceGetOrOpenDatasetEntryForTable(
      context, *this, display_uri, &result->dataset_cache_hit);
  result->dataset =
      result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
  result->file_path = display_uri;
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
  auto live = LanceTableSchema::FromArrow(
      context, result->schema_root.arrow_schema, result->arrow_table);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  result->json_columns = live.json_columns;
  result->blob_columns = live.blob_columns;

  ValidateLiveSchemaOrReplace(context, live, result->file_path);

  auto *scan_schema_handle = lance_get_schema_for_scan(result->dataset);
  if (!scan_schema_handle) {
    throw IOException("Failed to get scan schema from Lance dataset: " +
                      result->file_path + LanceFormatErrorSuffix());
  }
  memset(&result->scan_schema_root.arrow_schema, 0,
         sizeof(result->scan_schema_root.arrow_schema));
  if (lance_schema_to_arrow(scan_schema_handle,
                            &result->scan_schema_root.arrow_schema) != 0) {
    lance_free_schema(scan_schema_handle);
    throw IOException(
        "Failed to export Lance scan schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(scan_schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&result->scan_schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->scan_arrow_table, result->scan_schema_root.arrow_schema);

  bind_data = std::move(result);
  return LanceTableScanFunction();
}

void RegisterLanceScan(ExtensionLoader &loader) {
  TableFunction internal_scan("__lance_scan", {LogicalType::VARCHAR},
                              LanceScanFunc, LanceScanBind, LanceScanInitGlobal,
                              LanceScanLocalInit);

  internal_scan.GetSignature().WithTypedKwargs(
      "options", [](TypedKwargs &options) {
        options.Add("explain_verbose", LogicalType::BOOLEAN);
      });
  internal_scan.projection_pushdown = true;
  internal_scan.filter_pushdown = true;
  internal_scan.filter_prune = true;
  internal_scan.sampling_pushdown = true;
  internal_scan.statistics = LanceScanStatistics;
  internal_scan.cardinality = LanceScanCardinality;
  internal_scan.get_partition_stats = LanceScanGetPartitionStats;
  internal_scan.supports_pushdown_type = LanceSupportsPushdownType;
  internal_scan.pushdown_expression = LancePushdownExpression;
  internal_scan.pushdown_complex_filter = LancePushdownComplexFilter;
  internal_scan.pushdown_expression = LancePushdownExpression;
  internal_scan.get_virtual_columns = LanceGetVirtualColumns;
  internal_scan.to_string = LanceScanToString;
  internal_scan.get_metrics = LanceScanGetMetrics;

  CreateTableFunctionInfo scan_info(std::move(internal_scan));
  scan_info.internal = true;
  loader.RegisterFunction(std::move(scan_info));

  TableFunction internal_namespace_scan(
      "__lance_namespace_scan",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceScanFunc, LanceNamespaceScanBind, LanceScanInitGlobal,
      LanceScanLocalInit);

  internal_namespace_scan.GetSignature().WithTypedKwargs(
      "options", [](TypedKwargs &options) {
        options.Add("explain_verbose", LogicalType::BOOLEAN);
        options.Add("token", LogicalType::VARCHAR);
        options.Add("bearer_token", LogicalType::VARCHAR);
        options.Add("api_key", LogicalType::VARCHAR);
      });
  internal_namespace_scan.projection_pushdown = true;
  internal_namespace_scan.filter_pushdown = true;
  internal_namespace_scan.filter_prune = true;
  internal_namespace_scan.sampling_pushdown = true;
  internal_namespace_scan.statistics = LanceScanStatistics;
  internal_namespace_scan.cardinality = LanceScanCardinality;
  internal_namespace_scan.get_partition_stats = LanceScanGetPartitionStats;
  internal_namespace_scan.supports_pushdown_type = LanceSupportsPushdownType;
  internal_namespace_scan.pushdown_expression = LancePushdownExpression;
  internal_namespace_scan.pushdown_complex_filter = LancePushdownComplexFilter;
  internal_namespace_scan.pushdown_expression = LancePushdownExpression;
  internal_namespace_scan.get_virtual_columns = LanceGetVirtualColumns;
  internal_namespace_scan.to_string = LanceScanToString;
  internal_namespace_scan.get_metrics = LanceScanGetMetrics;

  CreateTableFunctionInfo internal_info(std::move(internal_namespace_scan));
  internal_info.internal = true;
  internal_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
  loader.RegisterFunction(std::move(internal_info));

  TableFunction exec_fun = LanceExecFunction();
  CreateTableFunctionInfo exec_info(std::move(exec_fun));
  exec_info.internal = true;
  loader.RegisterFunction(std::move(exec_info));
}

} // namespace duckdb
