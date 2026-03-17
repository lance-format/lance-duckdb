#include "duckdb.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception_format_value.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_binder/constant_binder.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_update.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_filter_ir.hpp"
#include "lance_expr_ir.hpp"
#include "lance_insert.hpp"
#include "lance_scan_bind_data.hpp"
#include "lance_table_entry.hpp"
#include "lance_update.hpp"

#include <cstdint>

namespace duckdb {

static vector<pair<string, string>> ParseLanceMetadataRows(const char *ptr) {
  if (!ptr) {
    throw IOException("Failed to read Lance field metadata" +
                      LanceFormatErrorSuffix());
  }

  string joined = ptr;
  lance_free_string(ptr);

  vector<pair<string, string>> out;
  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    auto parts = StringUtil::Split(line, '\t');
    if (parts.size() != 2) {
      continue;
    }
    out.emplace_back(std::move(parts[0]), std::move(parts[1]));
  }
  return out;
}

static bool TryGetLanceFieldDefaultExpr(ClientContext &context,
                                        LanceTableEntry &table,
                                        const string &field_name,
                                        string &out_default_expr) {
  out_default_expr.clear();

  string display_uri;
  auto *dataset = LanceOpenDatasetForTable(context, table, display_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset for UPDATE planning: " +
                      display_uri + LanceFormatErrorSuffix());
  }

  auto rows = ParseLanceMetadataRows(
      lance_dataset_list_field_metadata(dataset, field_name.c_str()));
  lance_close_dataset(dataset);

  for (auto &entry : rows) {
    if (entry.first == "duckdb_default_expr") {
      out_default_expr = std::move(entry.second);
      return true;
    }
  }
  return false;
}

static unique_ptr<Expression>
BindStoredDefaultExpr(ClientContext &context, const string &catalog_name,
                      const string &schema_name, const string &default_expr_sql,
                      const LogicalType &target_type) {
  auto expr_list =
      Parser::ParseExpressionList(default_expr_sql, context.GetParserOptions());
  if (expr_list.size() != 1 || !expr_list[0]) {
    throw InternalException(
        "Lance UPDATE stored default expression is invalid");
  }

  (void)catalog_name;
  (void)schema_name;
  auto binder = Binder::CreateBinder(context);
  ConstantBinder default_binder(*binder, context, "DEFAULT value");
  default_binder.target_type = target_type;
  return default_binder.Bind(expr_list[0]);
}

struct LanceUpdatePlanParts {
  LogicalProjection *projection = nullptr;
  LogicalGet *get = nullptr;
  vector<LogicalFilter *> filters;
};

static LanceUpdatePlanParts ExtractUpdatePlanParts(LogicalUpdate &op) {
  LanceUpdatePlanParts parts;
  if (op.children.empty() || !op.children[0]) {
    throw InternalException("Lance UPDATE expects a child plan");
  }

  parts.projection = dynamic_cast<LogicalProjection *>(op.children[0].get());
  if (!parts.projection) {
    throw NotImplementedException(
        "Lance UPDATE only supports the standard UPDATE planning path");
  }

  LogicalOperator *node = parts.projection->children.empty()
                              ? nullptr
                              : parts.projection->children[0].get();
  if (!node) {
    throw InternalException("Lance UPDATE projection has no child");
  }

  // Optimizer passes can introduce extra projections between UPDATE's
  // projection and the base scan/filter nodes (e.g. pruning, projection
  // pushdown). Walk down projection/filter layers until we reach the base scan.
  while (true) {
    if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      auto &proj = node->Cast<LogicalProjection>();
      if (proj.children.empty() || !proj.children[0]) {
        throw InternalException("Lance UPDATE projection has no child");
      }
      node = proj.children[0].get();
      continue;
    }
    if (node->type == LogicalOperatorType::LOGICAL_FILTER) {
      auto &filter = node->Cast<LogicalFilter>();
      parts.filters.push_back(&filter);
      if (filter.children.empty() || !filter.children[0]) {
        throw InternalException("Lance UPDATE filter has no child");
      }
      node = filter.children[0].get();
      continue;
    }
    break;
  }

  parts.get = dynamic_cast<LogicalGet *>(node);
  if (!parts.get) {
    throw NotImplementedException(
        "Lance UPDATE does not support UPDATE with joins or FROM");
  }

  return parts;
}

class PhysicalLanceUpdateOverwrite final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceUpdateOverwrite(PhysicalPlan &physical_plan,
                               vector<LogicalType> types_p,
                               LanceTableEntry &table_p, string predicate_ir_p,
                               vector<string> set_columns_p,
                               vector<string> set_expr_irs_p,
                               idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), predicate_ir(std::move(predicate_ir_p)),
        set_columns(std::move(set_columns_p)),
        set_expr_irs(std::move(set_expr_irs_p)) {}

  bool IsSource() const override { return true; }

  class UpdateSourceState : public GlobalSourceState {
  public:
    bool emitted = false;
    uint64_t rows_updated = 0;
  };

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<UpdateSourceState>();
  }

  SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                   OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<UpdateSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    if (set_columns.size() != set_expr_irs.size()) {
      throw InternalException("Lance UPDATE has mismatched SET columns/values");
    }

    string open_path;
    vector<string> option_keys;
    vector<string> option_values;
    string display_uri;
    ResolveLanceStorageOptionsForTable(context.client, table, open_path,
                                       option_keys, option_values, display_uri);

    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                    value_ptrs);

    vector<const char *> set_col_ptrs;
    vector<const uint8_t *> set_expr_ir_ptrs;
    vector<size_t> set_expr_ir_lens;
    set_col_ptrs.reserve(set_columns.size());
    set_expr_ir_ptrs.reserve(set_expr_irs.size());
    set_expr_ir_lens.reserve(set_expr_irs.size());
    for (idx_t i = 0; i < set_columns.size(); i++) {
      set_col_ptrs.push_back(set_columns[i].c_str());
      set_expr_ir_ptrs.push_back(
          reinterpret_cast<const uint8_t *>(set_expr_irs[i].data()));
      set_expr_ir_lens.push_back(set_expr_irs[i].size());
    }

    void *txn = nullptr;
    uint64_t rows_updated = 0;
    auto predicate_ptr =
        predicate_ir.empty()
            ? nullptr
            : reinterpret_cast<const uint8_t *>(predicate_ir.data());
    auto rc = lance_overwrite_update_transaction_with_irs_and_storage_options(
        open_path.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
        predicate_ptr, predicate_ir.size(), set_col_ptrs.data(),
        set_expr_ir_ptrs.data(), set_expr_ir_lens.data(), set_columns.size(),
        LANCE_DEFAULT_MAX_ROWS_PER_FILE, LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
        LANCE_DEFAULT_MAX_BYTES_PER_FILE, &txn, &rows_updated);
    if (rc != 0) {
      throw IOException("Failed to create Lance UPDATE transaction for '" +
                        open_path + "'" + LanceFormatErrorSuffix());
    }
    if (!txn) {
      if (rows_updated != 0) {
        throw IOException(
            "Failed to create Lance UPDATE transaction for '" + open_path +
            "': null transaction returned for non-zero updated rows");
      }
      state.rows_updated = 0;
      chunk.SetCardinality(1);
      chunk.SetValue(0, 0,
                     Value::BIGINT(NumericCast<int64_t>(state.rows_updated)));
      return SourceResultType::FINISHED;
    }

    RegisterLancePendingAppend(context.client, table.catalog,
                               std::move(open_path), std::move(option_keys),
                               std::move(option_values), txn);

    state.rows_updated = rows_updated;

    chunk.SetCardinality(1);
    chunk.SetValue(0, 0,
                   Value::BIGINT(NumericCast<int64_t>(state.rows_updated)));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceUpdateOverwrite"; }

  InsertionOrderPreservingMap<string> ParamsToString() const override {
    InsertionOrderPreservingMap<string> result;
    result["Predicate IR Bytes"] =
        predicate_ir.empty() ? "0" : to_string(predicate_ir.size());
    result["SET Columns"] = StringUtil::Join(set_columns, "\n");
    result["SET Expr IR Count"] = to_string(set_expr_irs.size());
    return result;
  }

private:
  LanceTableEntry &table;
  string predicate_ir;
  vector<string> set_columns;
  vector<string> set_expr_irs;
};

PhysicalOperator &PlanLanceUpdateOverwrite(ClientContext &context,
                                           PhysicalPlanGenerator &planner,
                                           LogicalUpdate &op) {
  if (op.return_chunk) {
    throw NotImplementedException("Lance UPDATE does not support RETURNING");
  }

  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException(
        "PlanLanceUpdateOverwrite called for non-Lance table");
  }

  auto parts = ExtractUpdatePlanParts(op);
  auto *scan_bind =
      parts.get && parts.get->bind_data
          ? dynamic_cast<LanceScanBindData *>(parts.get->bind_data.get())
          : nullptr;
  if (!scan_bind) {
    throw InternalException("Lance UPDATE is missing Lance scan bind data");
  }

  vector<string> predicate_ir_parts;
  if (!parts.get->table_filters.filters.empty() &&
      !TryBuildLanceTableFilterIRParts(scan_bind->names, scan_bind->types,
                                       parts.get->table_filters,
                                       predicate_ir_parts)) {
    throw NotImplementedException(
        "Lance UPDATE does not support one or more pushed table filters");
  }

  for (auto &part : scan_bind->lance_pushed_filter_ir_parts) {
    if (!part.empty()) {
      predicate_ir_parts.push_back(part);
    }
  }

  for (auto *filter : parts.filters) {
    if (!filter) {
      continue;
    }
    if (filter->children.empty() || !filter->children[0]) {
      throw InternalException("Lance UPDATE filter has no child");
    }
    for (auto &expr : filter->expressions) {
      if (!expr) {
        continue;
      }
      if (expr->IsFoldable()) {
        Value folded;
        if (ExpressionExecutor::TryEvaluateScalar(context, *expr, folded) &&
            folded.type() == LogicalTypeId::BOOLEAN && !folded.IsNull() &&
            folded.GetValue<bool>()) {
          continue;
        }
      }
      string expr_ir;
      if (!TryBuildLanceExprFilterIR(*parts.get, scan_bind->names,
                                     scan_bind->types, false, *expr, expr_ir)) {
        throw NotImplementedException(
            "Lance UPDATE does not support WHERE expression: %s",
            expr->ToString());
      }
      predicate_ir_parts.push_back(std::move(expr_ir));
    }
  }

  string predicate_ir;
  if (!TryEncodeLanceFilterIRMessage(predicate_ir_parts, predicate_ir)) {
    throw InternalException("Failed to encode Lance UPDATE predicate IR");
  }

  vector<string> set_columns;
  vector<string> set_expr_irs;
  set_columns.reserve(op.columns.size());
  set_expr_irs.reserve(op.columns.size());

  if (!parts.projection) {
    throw InternalException("Lance UPDATE is missing projection");
  }

  if (parts.projection->children.empty() || !parts.projection->children[0]) {
    throw InternalException("Lance UPDATE projection has no child");
  }
  auto &set_expr_scope = *parts.projection->children[0];

  for (idx_t i = 0; i < op.columns.size(); i++) {
    auto &target_col = op.table.GetColumns().GetColumn(op.columns[i]);
    set_columns.push_back(target_col.Name());

    if (!op.expressions[i]) {
      throw InternalException("Lance UPDATE has null SET expression");
    }

    if (op.expressions[i]->GetExpressionType() ==
        ExpressionType::VALUE_DEFAULT) {
      unique_ptr<Expression> target_default;
      string stored_default_expr;
      if (TryGetLanceFieldDefaultExpr(context, *lance_table, target_col.Name(),
                                      stored_default_expr)) {
        target_default = BindStoredDefaultExpr(
            context, op.table.catalog.GetName(), op.table.ParentSchema().name,
            stored_default_expr, target_col.Type());
      } else {
        auto storage_idx = target_col.StorageOid();
        if (storage_idx >= op.bound_defaults.size()) {
          throw InternalException(
              "Lance UPDATE default expression is missing for target column");
        }
        auto &bound_default = op.bound_defaults[storage_idx];
        if (!bound_default) {
          throw InternalException(
              "Lance UPDATE default expression is missing for target column");
        }
        target_default = bound_default->Copy();
      }
      string expr_ir;
      EncodeLanceExprIR(*parts.get, *scan_bind, set_expr_scope, *target_default,
                        expr_ir);
      set_expr_irs.push_back(std::move(expr_ir));
      continue;
    }

    unique_ptr<Expression> set_expr;
    if (op.expressions[i]->GetExpressionClass() == ExpressionClass::BOUND_REF) {
      auto &ref = op.expressions[i]->Cast<BoundReferenceExpression>();
      auto idx = NumericCast<idx_t>(ref.index);
      if (idx >= parts.projection->expressions.size()) {
        throw InternalException(
            "Lance UPDATE SET expression has invalid projection index");
      }
      set_expr = parts.projection->expressions[idx]->Copy();
    } else if (op.expressions[i]->GetExpressionClass() ==
               ExpressionClass::BOUND_COLUMN_REF) {
      auto &ref = op.expressions[i]->Cast<BoundColumnRefExpression>();
      if (ref.binding.table_index != parts.projection->table_index) {
        throw NotImplementedException(
            "Lance UPDATE does not support UPDATE with joins or FROM");
      }
      if (ref.binding.column_index >= parts.projection->expressions.size()) {
        throw InternalException(
            "Lance UPDATE SET expression has invalid projection index");
      }
      set_expr =
          parts.projection->expressions[ref.binding.column_index]->Copy();
    } else {
      set_expr = op.expressions[i]->Copy();
    }
    string expr_ir;
    EncodeLanceExprIR(*parts.get, *scan_bind, set_expr_scope, *set_expr,
                      expr_ir);
    set_expr_irs.push_back(std::move(expr_ir));
  }

  return planner.Make<PhysicalLanceUpdateOverwrite>(
      op.types, *lance_table, std::move(predicate_ir), std::move(set_columns),
      std::move(set_expr_irs), op.estimated_cardinality);
}

} // namespace duckdb
