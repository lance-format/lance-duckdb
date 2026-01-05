#include "duckdb.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/constraint.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/parser/parsed_data/comment_on_column_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/sample_options.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_filter_ir.hpp"
#include "lance_scan_bind_data.hpp"
#include "lance_table_entry.hpp"

#include <atomic>
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

static bool LancePushdownExpression(ClientContext &, const LogicalGet &,
                                    Expression &expr) {
  if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
    return true;
  }
  auto &func = expr.Cast<BoundFunctionExpression>();
  auto &name = func.function.name;
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

LanceScanBindData::~LanceScanBindData() {
  if (dataset) {
    lance_close_dataset(dataset);
  }
}

static constexpr column_t LANCE_COLUMN_IDENTIFIER_ROW_ID =
    UINT64_C(9223372036854775900);

static constexpr const char *LANCE_ROW_ID_COLUMN_NAME = "_rowid";

static bool IsLanceVirtualRowIdColumnId(column_t col_id) {
  return col_id == LANCE_COLUMN_IDENTIFIER_ROW_ID;
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
  size_t filter_len = 0;
  if (filter_ir && !filter_ir->empty()) {
    filter_ptr = reinterpret_cast<const uint8_t *>(filter_ir->data()); // NOLINT
    filter_len = filter_ir->size();
  }

  auto limit_i64 = pushed_limit.IsValid()
                       ? NumericCast<int64_t>(pushed_limit.GetIndex())
                       : int64_t(-1);
  auto offset_i64 = NumericCast<int64_t>(pushed_offset);

  auto *plan_ptr = lance_explain_dataset_scan_ir(
      dataset, col_ptrs.empty() ? nullptr : col_ptrs.data(), col_ptrs.size(),
      filter_ptr, filter_len, limit_i64, offset_i64, verbose ? 1 : 0);
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

  bool use_dataset_scanner = false;
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

  vector<column_t> scan_column_ids;
  vector<string> scan_column_names;
  string lance_filter_ir;
  bool filter_pushed_down = false;

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

  DataChunk scan_converted;
  void *stream = nullptr;
  LanceScanGlobalState *global_state = nullptr;
  idx_t fragment_pos = 0;
  bool filter_pushed_down = false;
  SelectionVector filter_sel;

  ~LanceScanLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static bool LanceSupportsPushdownType(const FunctionData &bind_data,
                                      idx_t col_idx) {
  auto &scan_bind = bind_data.Cast<LanceScanBindData>();
  if (col_idx >= scan_bind.types.size()) {
    return false;
  }
  return LanceFilterIRSupportsLogicalType(scan_bind.types[col_idx]);
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

static void CastRowIdToDuckDBRowType(Vector &src, Vector &dst, idx_t count) {
  if (count == 0) {
    return;
  }
  if (src.GetType() != LogicalType::UBIGINT ||
      dst.GetType() != LogicalType::ROW_TYPE) {
    throw InternalException("Invalid rowid cast types");
  }

  UnifiedVectorFormat format;
  src.ToUnifiedFormat(count, format);
  auto src_data = UnifiedVectorFormat::GetData<uint64_t>(format);

  dst.SetVectorType(VectorType::FLAT_VECTOR);
  auto dst_data = FlatVector::GetData<int64_t>(dst);
  auto &dst_validity = FlatVector::Validity(dst);
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

static bool TryExtractTakeRowIdsFromFilter(const TableFilter &filter,
                                           vector<uint64_t> &out_row_ids) {
  switch (filter.filter_type) {
  case TableFilterType::IN_FILTER: {
    auto &in = filter.Cast<InFilter>();
    out_row_ids.reserve(out_row_ids.size() + in.values.size());
    for (auto &v : in.values) {
      uint64_t row_id = 0;
      if (!TryParseRowIdValue(v, row_id)) {
        throw InvalidInputException("Lance point lookup requires non-negative "
                                    "integer rowid/_rowid values");
      }
      out_row_ids.push_back(row_id);
    }
    return true;
  }
  case TableFilterType::CONSTANT_COMPARISON: {
    auto &cmp = filter.Cast<ConstantFilter>();
    if (cmp.comparison_type != ExpressionType::COMPARE_EQUAL) {
      return false;
    }
    uint64_t row_id = 0;
    if (!TryParseRowIdValue(cmp.constant, row_id)) {
      throw InvalidInputException("Lance point lookup requires non-negative "
                                  "integer rowid/_rowid values");
    }
    out_row_ids.push_back(row_id);
    return true;
  }
  case TableFilterType::OPTIONAL_FILTER: {
    auto &opt = filter.Cast<OptionalFilter>();
    if (!opt.child_filter) {
      return false;
    }
    return TryExtractTakeRowIdsFromFilter(*opt.child_filter, out_row_ids);
  }
  default:
    return false;
  }
}

static bool IsTakeRowIdFilter(const TableFilter &filter) {
  switch (filter.filter_type) {
  case TableFilterType::IN_FILTER:
    return true;
  case TableFilterType::CONSTANT_COMPARISON: {
    auto &cmp = filter.Cast<ConstantFilter>();
    return cmp.comparison_type == ExpressionType::COMPARE_EQUAL;
  }
  case TableFilterType::OPTIONAL_FILTER: {
    auto &opt = filter.Cast<OptionalFilter>();
    if (!opt.child_filter) {
      return false;
    }
    return IsTakeRowIdFilter(*opt.child_filter);
  }
  default:
    return false;
  }
}

static bool TryExtractTakeRowIdsFromFilters(const TableFilterSet &filters,
                                            const idx_t row_id_col_idx,
                                            vector<uint64_t> &out_row_ids) {
  auto it = filters.filters.find(row_id_col_idx);
  if (it == filters.filters.end() || !it->second) {
    return false;
  }
  return TryExtractTakeRowIdsFromFilter(*it->second, out_row_ids);
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
    if (colref.binding.table_index != get.table_index ||
        colref.binding.column_index >= col_ids.size()) {
      return false;
    }
    auto col_id = col_ids[colref.binding.column_index].GetPrimaryIndex();
    return col_id == COLUMN_IDENTIFIER_ROW_ID ||
           IsLanceVirtualRowIdColumnId(col_id);
  };

  std::function<bool(const Expression &, vector<uint64_t> &)>
      try_extract_rowids;
  try_extract_rowids = [&](const Expression &expr, vector<uint64_t> &out) {
    if (expr.GetExpressionType() == ExpressionType::COMPARE_IN &&
        expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
      auto &op = expr.Cast<BoundOperatorExpression>();
      if (op.children.size() <= 1 || !op.children[0] ||
          op.children[0]->GetExpressionClass() !=
              ExpressionClass::BOUND_COLUMN_REF) {
        return false;
      }
      auto &colref = op.children[0]->Cast<BoundColumnRefExpression>();
      if (!is_rowid_ref(colref)) {
        return false;
      }
      out.clear();
      out.reserve(op.children.size() - 1);
      for (idx_t i = 1; i < op.children.size(); i++) {
        auto &child = op.children[i];
        if (!child ||
            child->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
          return false;
        }
        uint64_t row_id = 0;
        auto &v = child->Cast<BoundConstantExpression>().value;
        if (!TryParseRowIdValue(v, row_id)) {
          throw InvalidInputException(
              "Lance point lookup requires non-negative integer rowid/_rowid "
              "values");
        }
        out.push_back(row_id);
      }
      return true;
    }

    if (expr.GetExpressionType() == ExpressionType::COMPARE_EQUAL &&
        expr.GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
      auto &cmp = expr.Cast<BoundComparisonExpression>();
      if (!cmp.left || !cmp.right) {
        return false;
      }
      auto extract_one = [&](const Expression &lhs,
                             const Expression &rhs) -> bool {
        if (lhs.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
            rhs.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
          return false;
        }
        auto &colref = lhs.Cast<BoundColumnRefExpression>();
        if (!is_rowid_ref(colref)) {
          return false;
        }
        uint64_t row_id = 0;
        auto &v = rhs.Cast<BoundConstantExpression>().value;
        if (!TryParseRowIdValue(v, row_id)) {
          throw InvalidInputException(
              "Lance point lookup requires non-negative integer rowid/_rowid "
              "values");
        }
        out.clear();
        out.push_back(row_id);
        return true;
      };
      if (extract_one(*cmp.left, *cmp.right)) {
        return true;
      }
      if (extract_one(*cmp.right, *cmp.left)) {
        return true;
      }
      return false;
    }

    if (expr.GetExpressionType() == ExpressionType::CONJUNCTION_OR &&
        expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
      auto &conj = expr.Cast<BoundConjunctionExpression>();
      vector<uint64_t> child_out;
      out.clear();
      out.reserve(conj.children.size());
      for (auto &child : conj.children) {
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
    if (scan_bind.take_row_ids.empty()) {
      vector<uint64_t> take_row_ids;
      if (try_extract_rowids(*expr, take_row_ids) && !take_row_ids.empty()) {
        scan_bind.take_row_ids = std::move(take_row_ids);
      }
    }
    if (expr->expression_class == ExpressionClass::BOUND_COMPARISON) {
      auto &cmp = expr->Cast<BoundComparisonExpression>();
      if (cmp.type == ExpressionType::COMPARE_DISTINCT_FROM ||
          cmp.type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        auto is_constant = [](const unique_ptr<Expression> &node) -> bool {
          if (!node) {
            return false;
          }
          if (node->expression_class == ExpressionClass::BOUND_CONSTANT) {
            return true;
          }
          if (node->expression_class == ExpressionClass::BOUND_CAST) {
            auto &cast = node->Cast<BoundCastExpression>();
            return !cast.try_cast && cast.child &&
                   cast.child->expression_class ==
                       ExpressionClass::BOUND_CONSTANT;
          }
          return false;
        };

        auto is_column = [](const unique_ptr<Expression> &node) -> bool {
          if (!node) {
            return false;
          }
          return node->expression_class == ExpressionClass::BOUND_COLUMN_REF ||
                 node->expression_class == ExpressionClass::BOUND_REF;
        };

        // Prefer TableFilterSet + ExpressionFilter for the 1-column + constant
        // form (keeps DuckDB fallback available without duplicating the
        // predicate in LancePushedFilterParts).
        if ((is_column(cmp.left) && is_constant(cmp.right)) ||
            (is_column(cmp.right) && is_constant(cmp.left))) {
          continue;
        }
      }
    }
    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types, false,
                                   *expr, filter_ir)) {
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));

    auto expr_copy = expr->Copy();
    ExpressionIterator::EnumerateExpression(expr_copy, [&](Expression &node) {
      if (node.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return;
      }
      auto &colref = node.Cast<BoundColumnRefExpression>();
      if (colref.binding.table_index != get.table_index ||
          colref.binding.column_index >= get_column_names.size()) {
        throw NotImplementedException(
            "Lance scan filter pushdown does not support joins");
      }
      auto &name = get_column_names[colref.binding.column_index];
      if (name.empty()) {
        throw NotImplementedException(
            "Lance scan filter pushdown could not resolve column name");
      }
      colref.alias = quote_identifier(name);
    });
    auto sql_part = expr_copy->ToString();
    scan_bind.duckdb_pushed_filter_sql_parts.push_back(sql_part);
  }
}

static unique_ptr<FunctionData> LanceScanBind(ClientContext &context,
                                              TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types,
                                              vector<string> &names) {
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

  result->dataset = LanceOpenDataset(context, result->file_path);
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
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->scan_arrow_table, result->scan_schema_root.arrow_schema);
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<FunctionData>
LanceNamespaceScanBind(ClientContext &context, TableFunctionBindInput &input,
                       vector<LogicalType> &return_types,
                       vector<string> &names) {
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
  result->dataset = LanceOpenDatasetInNamespace(
      context, endpoint, table_id, bearer_token, api_key, delimiter, table_uri);
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

  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();

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
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->scan_arrow_table, result->scan_schema_root.arrow_schema);
  names = result->names;
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

  for (auto col_id : input.column_ids) {
    if (col_id == COLUMN_IDENTIFIER_EMPTY) {
      scan_state.scanned_types.push_back(LogicalType::BOOLEAN);
      scan_state.output_to_scan_converted_idx.push_back(
          DConstants::INVALID_INDEX);
      scan_state.output_is_duckdb_rowid.push_back(false);
      scan_state.output_is_empty.push_back(true);
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
      continue;
    }

    if (col_id >= bind_data.types.size() || col_id >= bind_data.names.size()) {
      throw IOException("Invalid column id in projection");
    }
    auto scan_idx = add_scan_column(col_id, bind_data.names[col_id],
                                    bind_data.types[col_id]);
    scan_state.scanned_types.push_back(bind_data.types[col_id]);
    scan_state.output_to_scan_converted_idx.push_back(scan_idx);
    scan_state.output_is_duckdb_rowid.push_back(false);
    scan_state.output_is_empty.push_back(false);
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
        bind_data.names, bind_data.types, input, false);
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
      (!input.filters || input.filters->filters.empty()) &&
      scan_state.lance_filter_ir.empty()) {
    auto rows = lance_dataset_count_rows(bind_data.dataset);
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

  if (!scan_state.sampling_pushed_down && bind_data.limit_offset_pushed_down) {
    // Limit/offset pushdown requires that any TableFilterSet predicates are
    // evaluated by Lance. Otherwise limit/offset would apply before filtering.
    if (input.filters && !input.filters->filters.empty() &&
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
  result->column_ids = scan_global.scan_column_ids;
  result->filters = input.filters.get();
  result->global_state = &scan_global;
  result->filter_pushed_down = scan_global.filter_pushed_down;
  if (scan_global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, scan_global.scanned_types);
  }
  result->scan_converted.Initialize(context.client,
                                    scan_global.scan_converted_types);
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
  if (global_state.sampling_pushed_down) {
    local_state.filter_pushed_down = false;
    stream = lance_create_dataset_sample_stream_ir(
        bind_data.dataset, columns.data(), columns.size(),
        global_state.sample_percentage, global_state.sample_seed,
        global_state.sample_repeatable ? 1 : 0);
  } else if (global_state.use_dataset_take) {
    auto row_ids_ptr = global_state.take_row_ids.empty()
                           ? nullptr
                           : global_state.take_row_ids.data();
    stream = lance_create_dataset_take_stream(bind_data.dataset, row_ids_ptr,
                                              global_state.take_row_ids.size(),
                                              columns.data(), columns.size());
  } else if (global_state.use_dataset_scanner) {
    auto limit_i64 =
        global_state.pushed_limit.IsValid()
            ? NumericCast<int64_t>(global_state.pushed_limit.GetIndex())
            : int64_t(-1);
    auto offset_i64 = NumericCast<int64_t>(global_state.pushed_offset);
    stream = lance_create_dataset_stream_ir(
        bind_data.dataset, columns.data(), columns.size(), filter_ir,
        filter_ir_len, limit_i64, offset_i64);
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
      stream = lance_create_dataset_stream_ir(bind_data.dataset, columns.data(),
                                              columns.size(), nullptr, 0,
                                              limit_i64, offset_i64);
    }
  } else {
    if (local_state.fragment_pos >= global_state.fragment_ids.size()) {
      return false;
    }
    auto fragment_id = global_state.fragment_ids[local_state.fragment_pos];

    stream = lance_create_fragment_stream_ir(bind_data.dataset, fragment_id,
                                             columns.data(), columns.size(),
                                             filter_ir, filter_ir_len);
    if (!stream && filter_ir) {
      // Best-effort: if filter pushdown failed, retry without it and rely on
      // DuckDB-side filter execution for correctness.
      global_state.filter_pushdown_fallbacks.fetch_add(1);
      local_state.filter_pushed_down = false;
      stream = lance_create_fragment_stream_ir(bind_data.dataset, fragment_id,
                                               columns.data(), columns.size(),
                                               nullptr, 0);
    }
  }
  if (!stream) {
    throw IOException("Failed to create Lance scan stream" +
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

  if (tmp_schema.release) {
    tmp_schema.release(&tmp_schema);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static void LanceScanFunc(ClientContext &context, TableFunctionInput &data,
                          DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceScanBindData>();
  auto &global_state = data.global_state->Cast<LanceScanGlobalState>();
  auto &local_state = data.local_state->Cast<LanceScanLocalState>();
  auto &arrow_columns = global_state.scan_includes_virtual_rowid
                            ? bind_data.scan_arrow_table.GetColumns()
                            : bind_data.arrow_table.GetColumns();

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
    auto start = global_state.lines_read.fetch_add(output_size);

    local_state.scan_converted.Reset();
    local_state.scan_converted.SetCardinality(output_size);
    ArrowTableFunction::ArrowToDuckDB(local_state, arrow_columns,
                                      local_state.scan_converted, start);

    auto fill_output_from_converted = [&](DataChunk &target) {
      if (target.ColumnCount() !=
          global_state.output_to_scan_converted_idx.size()) {
        throw InternalException(
            "Lance scan output column count does not match mapping");
      }
      target.SetCardinality(output_size);
      for (idx_t i = 0; i < target.ColumnCount(); i++) {
        if (global_state.output_is_empty[i]) {
          target.data[i].Reference(Value::BOOLEAN(true));
          continue;
        }
        auto scan_idx = global_state.output_to_scan_converted_idx[i];
        if (scan_idx == DConstants::INVALID_INDEX ||
            scan_idx >= local_state.scan_converted.ColumnCount()) {
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
    };

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      local_state.all_columns.SetCardinality(output_size);
      fill_output_from_converted(local_state.all_columns);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      fill_output_from_converted(output);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
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
  if (TryLanceExplainDatasetScan(
          bind_data.dataset, nullptr,
          filter_ir_msg.empty() ? nullptr : &filter_ir_msg,
          bind_data.pushed_limit, bind_data.pushed_offset,
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
  result["Lance Scan Mode"] =
      global_state.use_dataset_scanner ? "dataset" : "fragment";
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

  return result;
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

static bool IsLanceScanTableFunction(const TableFunction &fn) {
  return fn.name == "__lance_scan" || fn.name == "__lance_table_scan" ||
         fn.name == "__lance_namespace_scan";
}

static bool IsLanceRowIdColumn(const LogicalGet &get,
                               const LanceScanBindData &scan_bind,
                               const BoundColumnRefExpression &colref) {
  if (colref.binding.table_index != get.table_index) {
    return false;
  }
  auto &col_ids = get.GetColumnIds();
  if (colref.binding.column_index >= col_ids.size()) {
    return false;
  }

  auto col_id = col_ids[colref.binding.column_index].GetPrimaryIndex();
  (void)scan_bind;
  return col_id == COLUMN_IDENTIFIER_ROW_ID ||
         IsLanceVirtualRowIdColumnId(col_id);
}

static bool IsLanceRowIdColumn(const LogicalGet &get,
                               const LanceScanBindData &scan_bind,
                               const BoundReferenceExpression &ref) {
  auto &col_ids = get.GetColumnIds();
  auto idx = NumericCast<idx_t>(ref.index);
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
    if (op.children.size() <= 1 || !op.children[0] ||
        !is_row_id(*op.children[0])) {
      return false;
    }

    out_row_ids.clear();
    out_row_ids.reserve(op.children.size() - 1);
    for (idx_t i = 1; i < op.children.size(); i++) {
      auto &child = op.children[i];
      if (!child ||
          child->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
        return false;
      }
      uint64_t row_id = 0;
      parse_row_id(child->Cast<BoundConstantExpression>().value, row_id);
      out_row_ids.push_back(row_id);
    }
    return true;
  }

  auto try_extract_equal = [&](const Expression &candidate, uint64_t &out) {
    if (candidate.GetExpressionType() != ExpressionType::COMPARE_EQUAL ||
        candidate.GetExpressionClass() != ExpressionClass::BOUND_COMPARISON) {
      return false;
    }
    auto &cmp = candidate.Cast<BoundComparisonExpression>();
    if (!cmp.left || !cmp.right) {
      return false;
    }

    if (is_row_id(*cmp.left) &&
        cmp.right->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
      parse_row_id(cmp.right->Cast<BoundConstantExpression>().value, out);
      return true;
    }
    if (is_row_id(*cmp.right) &&
        cmp.left->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
      parse_row_id(cmp.left->Cast<BoundConstantExpression>().value, out);
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
        for (auto it = conj.children.rbegin(); it != conj.children.rend();
             ++it) {
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

    column_t filter_col_id = LANCE_COLUMN_IDENTIFIER_ROW_ID;
    auto it = get.table_filters.filters.find(filter_col_id);
    if (it == get.table_filters.filters.end() || !it->second) {
      filter_col_id = COLUMN_IDENTIFIER_ROW_ID;
      it = get.table_filters.filters.find(filter_col_id);
    }
    if (it == get.table_filters.filters.end() || !it->second) {
      return op;
    }

    vector<uint64_t> row_ids;
    bool can_take = scan_bind.take_row_ids.empty() &&
                    TryExtractTakeRowIdsFromFilter(*it->second, row_ids) &&
                    !row_ids.empty();
    if (can_take) {
      scan_bind.take_row_ids = std::move(row_ids);
      get.table_filters.filters.erase(it);
      return op;
    }

    auto &col_ids = get.GetColumnIds();
    optional_idx col_pos = optional_idx::Invalid();
    for (idx_t i = 0; i < col_ids.size(); i++) {
      if (col_ids[i].GetPrimaryIndex() == filter_col_id) {
        col_pos = optional_idx(i);
        break;
      }
    }
    if (!col_pos.IsValid()) {
      throw InternalException(
          "Lance scan found a rowid table filter without a rowid column");
    }

    auto col_type = filter_col_id == COLUMN_IDENTIFIER_ROW_ID
                        ? LogicalType::ROW_TYPE
                        : LogicalType::UBIGINT;
    auto colref = make_uniq<BoundColumnRefExpression>(
        col_type, ColumnBinding(get.table_index, col_pos.GetIndex()));
    auto filter_expr = it->second->ToExpression(*colref);
    get.table_filters.filters.erase(it);

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
  if (get.extra_info.sample_options) {
    // Sampling must occur before limiting, so do not remove the LIMIT node.
    return op;
  }

  auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();
  if (!scan_bind.take_row_ids.empty()) {
    return op;
  }
  auto &col_ids = get.GetColumnIds();
  for (auto &entry : get.table_filters.filters) {
    if (!entry.second || entry.first >= col_ids.size()) {
      continue;
    }
    auto col_id = col_ids[entry.first].GetPrimaryIndex();
    if (col_id != COLUMN_IDENTIFIER_ROW_ID &&
        !IsLanceVirtualRowIdColumnId(col_id)) {
      continue;
    }
    if (IsTakeRowIdFilter(*entry.second)) {
      return op;
    }
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
    auto &name = func.function.name;
    if (name != "~~" && name != "~~*" && name != "like_escape" &&
        name != "ilike_escape") {
      continue;
    }

    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types, false,
                                   *expr, filter_ir)) {
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
    if (get.function.name != "__lance_scan" || get.parameters.size() != 1 ||
        get.parameters[0].IsNull() ||
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
  OptimizerExtension rowid_take_ext;
  rowid_take_ext.optimize_function = LanceRowIdInRewriteOptimizer;
  config.optimizer_extensions.push_back(std::move(rowid_take_ext));

  OptimizerExtension like_ext;
  like_ext.optimize_function = LanceLikePushdownOptimizer;
  config.optimizer_extensions.push_back(std::move(like_ext));

  OptimizerExtension limit_ext;
  limit_ext.optimize_function = LanceLimitOffsetPushdownOptimizer;
  config.optimizer_extensions.push_back(std::move(limit_ext));

  OptimizerExtension cardinality_ext;
  cardinality_ext.optimize_function = LanceCardinalityFixupOptimizer;
  config.optimizer_extensions.push_back(std::move(cardinality_ext));
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
  function.dynamic_to_string = LanceScanDynamicToString;
  function.init_global = LanceScanInitGlobal;
  function.init_local = LanceScanLocalInit;
  return function;
}

LanceTableEntry::LanceTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                 CreateTableInfo &info, string dataset_uri)
    : TableCatalogEntry(catalog, schema, info),
      dataset_uri(std::move(dataset_uri)) {}

LanceTableEntry::LanceTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                 CreateTableInfo &info,
                                 LanceNamespaceTableConfig config)
    : TableCatalogEntry(catalog, schema, info),
      dataset_uri(config.endpoint + "/" + config.table_id),
      namespace_config(
          make_uniq<LanceNamespaceTableConfig>(std::move(config))) {}

static unordered_map<string, string> ParseTsvKvs(const char *ptr) {
  unordered_map<string, string> out;
  if (!ptr) {
    return out;
  }

  string joined = ptr;
  lance_free_string(ptr);

  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    auto parts = StringUtil::Split(line, '\t');
    if (parts.size() != 2) {
      continue;
    }
    out[std::move(parts[0])] = std::move(parts[1]);
  }
  return out;
}

static void PopulateLanceTableSchemaFromDataset(
    ClientContext &context, void *dataset, ColumnList &out_columns,
    vector<unique_ptr<Constraint>> &out_constraints) {
  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    throw IOException("Failed to get schema from Lance dataset" +
                      LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);

  auto &config = DBConfig::GetConfig(context);
  ArrowTableSchema arrow_table;
  ArrowTableFunction::PopulateArrowTableSchema(config, arrow_table,
                                               schema_root.arrow_schema);
  const auto names = arrow_table.GetNames();
  const auto types = arrow_table.GetTypes();
  if (names.size() != types.size()) {
    throw InternalException(
        "Arrow table schema returned mismatched names/types sizes");
  }

  out_columns = ColumnList();
  out_constraints.clear();
  for (idx_t i = 0; i < names.size(); i++) {
    ColumnDefinition col(names[i], types[i]);
    auto *field_md =
        lance_dataset_list_field_metadata(dataset, names[i].c_str());
    if (!field_md) {
      throw IOException("Failed to list field metadata from Lance dataset" +
                        LanceFormatErrorSuffix());
    }
    auto kvs = ParseTsvKvs(field_md);
    auto it = kvs.find("comment");
    if (it != kvs.end()) {
      col.SetComment(Value(it->second));
    }
    out_columns.AddColumn(std::move(col));

    // Reflect not-null constraints for better DuckDB-side UX.
    auto *child = schema_root.arrow_schema.children[i];
    if (child && (child->flags & ARROW_FLAG_NULLABLE) == 0) {
      out_constraints.push_back(make_uniq<NotNullConstraint>(LogicalIndex(i)));
    }
  }
}

static unique_ptr<CatalogEntry>
BuildUpdatedLanceTableEntry(ClientContext &context, const LanceTableEntry &base,
                            bool internal) {
  string display_uri;
  void *dataset = LanceOpenDatasetForTable(context, base, display_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  auto &catalog = base.catalog;
  auto &schema = base.schema;

  CreateTableInfo create_info(schema, base.name);
  create_info.internal = internal;
  create_info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

  try {
    PopulateLanceTableSchemaFromDataset(context, dataset, create_info.columns,
                                        create_info.constraints);
  } catch (...) {
    lance_close_dataset(dataset);
    throw;
  }

  unique_ptr<LanceTableEntry> entry;
  if (base.IsNamespaceBacked()) {
    entry = make_uniq<LanceTableEntry>(catalog, schema, create_info,
                                       base.NamespaceConfig());
  } else {
    entry = make_uniq<LanceTableEntry>(catalog, schema, create_info,
                                       base.DatasetUri());
  }

  auto *table_md = lance_dataset_list_table_metadata(dataset);
  if (!table_md) {
    lance_close_dataset(dataset);
    throw IOException("Failed to list table metadata from Lance dataset" +
                      LanceFormatErrorSuffix());
  }
  auto table_kvs = ParseTsvKvs(table_md);
  auto it = table_kvs.find("comment");
  if (it != table_kvs.end()) {
    entry->comment = Value(it->second);
  }

  lance_close_dataset(dataset);
  return unique_ptr_cast<LanceTableEntry, CatalogEntry>(std::move(entry));
}

static void ValidateAlterColumnTypeTarget(const LogicalType &type) {
  switch (type.id()) {
  case LogicalTypeId::BOOLEAN:
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::BIGINT:
  case LogicalTypeId::UBIGINT:
  case LogicalTypeId::FLOAT:
  case LogicalTypeId::DOUBLE:
  case LogicalTypeId::DATE:
  case LogicalTypeId::TIME:
  case LogicalTypeId::TIMESTAMP:
  case LogicalTypeId::TIMESTAMP_TZ:
  case LogicalTypeId::VARCHAR:
  case LogicalTypeId::BLOB:
    return;
  default:
    break;
  }
  throw NotImplementedException(
      "Lance ALTER COLUMN TYPE only supports a limited set of DuckDB types "
      "(BOOLEAN, integer/floating, DATE/TIME/TIMESTAMP, VARCHAR, BLOB).");
}

static bool IsImplicitCastUsingExpression(const ParsedExpression &expr,
                                          const string &column_name,
                                          const LogicalType &target_type) {
  auto *cast_expr = dynamic_cast<const CastExpression *>(&expr);
  if (!cast_expr || cast_expr->try_cast ||
      cast_expr->cast_type != target_type) {
    return false;
  }
  auto *col_ref =
      dynamic_cast<const ColumnRefExpression *>(cast_expr->child.get());
  if (!col_ref || col_ref->column_names.size() != 1) {
    return false;
  }
  return StringUtil::CIEquals(col_ref->column_names[0], column_name);
}

unique_ptr<CatalogEntry>
LanceTableEntry::AlterEntry(CatalogTransaction transaction, AlterInfo &info) {
  if (!transaction.context) {
    throw InternalException(
        "LanceTableEntry::AlterEntry missing client context");
  }
  return AlterEntry(*transaction.context, info);
}

unique_ptr<CatalogEntry> LanceTableEntry::AlterEntry(ClientContext &context,
                                                     AlterInfo &info) {
  if (!context.transaction.IsAutoCommit()) {
    throw NotImplementedException(
        "Lance DDL does not support explicit transactions yet");
  }

  string display_uri;
  void *dataset = LanceOpenDatasetForTable(context, *this, display_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  unique_ptr<CatalogEntry> result;
  switch (info.type) {
  case AlterType::ALTER_TABLE: {
    auto &alter = info.Cast<AlterTableInfo>();
    switch (alter.alter_table_type) {
    case AlterTableType::ADD_COLUMN: {
      auto &add = info.Cast<AddColumnInfo>();
      vector<string> names{add.new_column.Name()};
      vector<LogicalType> types{add.new_column.Type()};

      ArrowSchemaWrapper new_schema_root;
      memset(&new_schema_root.arrow_schema, 0,
             sizeof(new_schema_root.arrow_schema));
      auto props = context.GetClientProperties();
      ArrowConverter::ToArrowSchema(&new_schema_root.arrow_schema, types, names,
                                    props);

      vector<string> expressions;
      if (add.new_column.HasDefaultValue()) {
        expressions.push_back(add.new_column.DefaultValue().ToString());
      }
      vector<const char *> expr_ptrs;
      expr_ptrs.reserve(expressions.size());
      for (auto &e : expressions) {
        expr_ptrs.push_back(e.c_str());
      }

      auto rc = lance_dataset_add_columns(
          dataset, &new_schema_root.arrow_schema,
          expr_ptrs.empty() ? nullptr : expr_ptrs.data(), expr_ptrs.size(), 0);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to add column to Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }

      // Best-effort persistence of DuckDB column defaults in Lance field
      // metadata. Re-open the dataset after schema evolution so the new field
      // is visible to the metadata updater.
      lance_close_dataset(dataset);
      dataset = nullptr;
      if (add.new_column.HasDefaultValue()) {
        string reopen_uri;
        dataset = LanceOpenDatasetForTable(context, *this, reopen_uri);
        if (dataset) {
          auto default_expr = add.new_column.DefaultValue().ToString();
          (void)lance_dataset_update_field_metadata(
              dataset, add.new_column.Name().c_str(), "duckdb_default_expr",
              default_expr.c_str());
          lance_close_dataset(dataset);
          dataset = nullptr;
        }
      } else {
        // The dataset was already closed above, keep going.
      }
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::REMOVE_COLUMN: {
      auto &drop = info.Cast<RemoveColumnInfo>();
      if (drop.if_column_exists && !ColumnExists(drop.removed_column)) {
        lance_close_dataset(dataset);
        result = nullptr;
        break;
      }
      const char *cols[1] = {drop.removed_column.c_str()};
      auto rc = lance_dataset_drop_columns(dataset, cols, 1);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to drop column from Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::RENAME_COLUMN: {
      auto &rename = info.Cast<RenameColumnInfo>();
      auto rc = lance_dataset_alter_columns_rename(
          dataset, rename.old_name.c_str(), rename.new_name.c_str());
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to rename column in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::ALTER_COLUMN_TYPE: {
      auto &cast = info.Cast<ChangeColumnTypeInfo>();
      ValidateAlterColumnTypeTarget(cast.target_type);
      if (!cast.expression ||
          !IsImplicitCastUsingExpression(*cast.expression, cast.column_name,
                                         cast.target_type)) {
        lance_close_dataset(dataset);
        throw NotImplementedException(
            "Lance ALTER COLUMN TYPE only supports implicit USING (i.e., a "
            "simple CAST of the original column)");
      }

      ArrowSchemaWrapper new_type_schema;
      memset(&new_type_schema.arrow_schema, 0,
             sizeof(new_type_schema.arrow_schema));
      auto props = context.GetClientProperties();
      ArrowConverter::ToArrowSchema(&new_type_schema.arrow_schema,
                                    {cast.target_type}, {cast.column_name},
                                    props);

      auto rc = lance_dataset_alter_columns_cast(
          dataset, cast.column_name.c_str(), &new_type_schema.arrow_schema);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to change column type in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::SET_NOT_NULL: {
      auto &nn = info.Cast<SetNotNullInfo>();
      auto rc = lance_dataset_alter_columns_set_nullable(
          dataset, nn.column_name.c_str(), 0);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to set NOT NULL in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::DROP_NOT_NULL: {
      auto &nn = info.Cast<DropNotNullInfo>();
      auto rc = lance_dataset_alter_columns_set_nullable(
          dataset, nn.column_name.c_str(), 1);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to drop NOT NULL in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    default:
      lance_close_dataset(dataset);
      throw NotImplementedException(
          "ALTER TABLE operation not supported for Lance tables");
    }
    break;
  }
  case AlterType::SET_COLUMN_COMMENT: {
    auto &comment = info.Cast<SetColumnCommentInfo>();
    const char *comment_ptr = nullptr;
    string comment_str;
    if (!comment.comment_value.IsNull()) {
      comment_str = comment.comment_value.DefaultCastAs(LogicalType::VARCHAR)
                        .GetValue<string>();
      comment_ptr = comment_str.c_str();
    }
    auto rc = lance_dataset_update_field_metadata(
        dataset, comment.column_name.c_str(), "comment", comment_ptr);
    if (rc != 0) {
      lance_close_dataset(dataset);
      throw IOException("Failed to update column comment in Lance dataset: " +
                        display_uri + LanceFormatErrorSuffix());
    }
    lance_close_dataset(dataset);
    return BuildUpdatedLanceTableEntry(context, *this, internal);
    break;
  }
  default:
    lance_close_dataset(dataset);
    throw NotImplementedException("ALTER is not supported for Lance tables");
  }

  return result;
}

unique_ptr<CatalogEntry> LanceTableEntry::Copy(ClientContext &context) const {
  (void)context;
  auto &catalog = this->catalog;
  auto &schema = this->schema;
  auto create_info = make_uniq<CreateTableInfo>(schema, name);
  create_info->temporary = temporary;
  create_info->internal = internal;
  create_info->comment = comment;
  create_info->tags = tags;
  create_info->columns = columns.Copy();
  for (auto &c : constraints) {
    create_info->constraints.push_back(c->Copy());
  }
  if (IsNamespaceBacked()) {
    return make_uniq_base<CatalogEntry, LanceTableEntry>(
        catalog, schema, *create_info, NamespaceConfig());
  }
  return make_uniq_base<CatalogEntry, LanceTableEntry>(
      catalog, schema, *create_info, dataset_uri);
}

TableFunction
LanceTableEntry::GetScanFunction(ClientContext &context,
                                 unique_ptr<FunctionData> &bind_data) {
  auto result = make_uniq<LanceScanBindData>();
  result->file_path = dataset_uri;

  string display_uri;
  result->dataset = LanceOpenDatasetForTable(context, *this, display_uri);
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

  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();

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
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->scan_arrow_table, result->scan_schema_root.arrow_schema);

  bind_data = std::move(result);
  return LanceTableScanFunction();
}

void RegisterLanceScan(ExtensionLoader &loader) {
  TableFunction internal_scan("__lance_scan", {LogicalType::VARCHAR},
                              LanceScanFunc, LanceScanBind, LanceScanInitGlobal,
                              LanceScanLocalInit);
  internal_scan.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
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
  internal_scan.dynamic_to_string = LanceScanDynamicToString;

  CreateTableFunctionInfo scan_info(std::move(internal_scan));
  scan_info.internal = true;
  loader.RegisterFunction(std::move(scan_info));

  TableFunction internal_namespace_scan(
      "__lance_namespace_scan",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceScanFunc, LanceNamespaceScanBind, LanceScanInitGlobal,
      LanceScanLocalInit);
  internal_namespace_scan.named_parameters["explain_verbose"] =
      LogicalType::BOOLEAN;
  internal_namespace_scan.named_parameters["token"] = LogicalType::VARCHAR;
  internal_namespace_scan.named_parameters["bearer_token"] =
      LogicalType::VARCHAR;
  internal_namespace_scan.named_parameters["api_key"] = LogicalType::VARCHAR;
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
  internal_namespace_scan.dynamic_to_string = LanceScanDynamicToString;

  CreateTableFunctionInfo internal_info(std::move(internal_namespace_scan));
  internal_info.internal = true;
  internal_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
  loader.RegisterFunction(std::move(internal_info));
}

} // namespace duckdb
