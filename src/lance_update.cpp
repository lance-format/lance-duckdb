#include "duckdb.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception_format_value.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_update.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_scan_bind_data.hpp"
#include "lance_table_entry.hpp"
#include "lance_update.hpp"

#include <cstdint>

namespace duckdb {

static string QuoteIdentifierForDataFusion(const string &name) {
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
}

static vector<string> BuildGetColumnNames(const LogicalGet &get,
                                          const TableCatalogEntry &table) {
  vector<string> names;
  auto &col_ids = get.GetColumnIds();
  names.reserve(col_ids.size());
  for (idx_t i = 0; i < col_ids.size(); i++) {
    auto col_id = col_ids[i].GetPrimaryIndex();
    if (col_id == COLUMN_IDENTIFIER_ROW_ID) {
      names.push_back("rowid");
      continue;
    }
    if (col_id == COLUMN_IDENTIFIER_EMPTY) {
      names.push_back("");
      continue;
    }
    if (col_id >= table.GetColumns().LogicalColumnCount()) {
      throw InternalException("Invalid column id in LogicalGet");
    }
    auto &col = table.GetColumn(LogicalIndex(NumericCast<idx_t>(col_id)));
    names.push_back(col.Name());
  }
  return names;
}

static vector<string> BuildGetOutputNames(const LogicalGet &get,
                                          const TableCatalogEntry &table);

static void SetAliasesForDataFusion(unique_ptr<Expression> &expr,
                                    const vector<string> &input_names) {
  ExpressionIterator::VisitExpressionMutable<BoundReferenceExpression>(
      expr, [&](BoundReferenceExpression &ref, unique_ptr<Expression> &) {
        auto idx = NumericCast<idx_t>(ref.index);
        if (idx >= input_names.size()) {
          throw InternalException("Bound reference has invalid index");
        }
        auto &name = input_names[idx];
        if (name.empty()) {
          throw NotImplementedException(
              "Lance UPDATE does not support expressions referencing COUNT(*)");
        }
        if (name == "rowid") {
          throw NotImplementedException(
              "Lance UPDATE does not support expressions referencing rowid");
        }
        ref.alias = QuoteIdentifierForDataFusion(name);
      });

  ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
      expr, [&](BoundColumnRefExpression &colref, unique_ptr<Expression> &) {
        if (colref.binding.column_index >= input_names.size()) {
          throw InternalException("Bound column ref has invalid column index");
        }
        auto &name = input_names[colref.binding.column_index];
        if (name.empty()) {
          throw NotImplementedException(
              "Lance UPDATE does not support expressions referencing COUNT(*)");
        }
        if (name == "rowid") {
          throw NotImplementedException(
              "Lance UPDATE does not support expressions referencing rowid");
        }
        colref.alias = QuoteIdentifierForDataFusion(name);
      });
}

static string StringifyExprForDataFusion(const Expression &expr,
                                         LogicalOperator &context_in,
                                         const TableCatalogEntry &table) {
  auto copy = expr.Copy();
  auto *context = &context_in;
  // Expand BoundReference/BoundColumnRef references through projection/filter
  // layers until we reach the base scan, then alias base columns using Lance's
  // SQL dialect quoting.
  while (true) {
    switch (context->type) {
    case LogicalOperatorType::LOGICAL_GET: {
      auto input_names =
          BuildGetOutputNames(context->Cast<LogicalGet>(), table);
      SetAliasesForDataFusion(copy, input_names);
      return copy->ToString();
    }
    case LogicalOperatorType::LOGICAL_FILTER: {
      auto &filter = context->Cast<LogicalFilter>();
      if (filter.children.empty() || !filter.children[0]) {
        throw InternalException("Lance UPDATE filter has no child");
      }
      context = filter.children[0].get();
      break;
    }
    case LogicalOperatorType::LOGICAL_PROJECTION: {
      auto &proj = context->Cast<LogicalProjection>();
      if (proj.children.empty() || !proj.children[0]) {
        throw InternalException("Lance UPDATE projection has no child");
      }

      ExpressionIterator::VisitExpressionMutable<BoundReferenceExpression>(
          copy,
          [&](BoundReferenceExpression &ref, unique_ptr<Expression> &node) {
            auto idx = NumericCast<idx_t>(ref.index);
            if (idx >= proj.expressions.size() || !proj.expressions[idx]) {
              throw InternalException(
                  "Lance UPDATE projection reference has invalid index");
            }
            node = proj.expressions[idx]->Copy();
          });

      ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
          copy,
          [&](BoundColumnRefExpression &ref, unique_ptr<Expression> &node) {
            if (ref.binding.table_index != proj.table_index) {
              throw NotImplementedException(
                  "Lance UPDATE does not support UPDATE with joins or FROM");
            }
            if (ref.binding.column_index >= proj.expressions.size() ||
                !proj.expressions[ref.binding.column_index]) {
              throw InternalException(
                  "Lance UPDATE projection reference has invalid column index");
            }
            node = proj.expressions[ref.binding.column_index]->Copy();
          });

      context = proj.children[0].get();
      break;
    }
    default:
      throw NotImplementedException(
          "Lance UPDATE does not support UPDATE with joins or FROM");
    }
  }
}

static string
StringifyTableFiltersForDataFusion(const TableFilterSet &filters,
                                   const TableCatalogEntry &table) {
  string predicate;
  for (auto &it : filters.filters) {
    if (!it.second) {
      continue;
    }
    auto col_id = it.first;
    if (col_id == COLUMN_IDENTIFIER_ROW_ID ||
        col_id == COLUMN_IDENTIFIER_EMPTY) {
      throw NotImplementedException(
          "Lance UPDATE does not support predicates on rowid/COUNT(*)");
    }
    if (col_id >= table.GetColumns().LogicalColumnCount()) {
      throw InternalException("Invalid table filter column id");
    }
    auto &col = table.GetColumn(LogicalIndex(NumericCast<idx_t>(col_id)));
    auto col_name = QuoteIdentifierForDataFusion(col.Name());
    auto part = it.second->ToString(col_name);
    if (predicate.empty()) {
      predicate = "(" + part + ")";
    } else {
      predicate += " AND (" + part + ")";
    }
  }
  return predicate;
}

static vector<string> BuildGetOutputNames(const LogicalGet &get,
                                          const TableCatalogEntry &table) {
  auto all_names = BuildGetColumnNames(get, table);
  if (get.projection_ids.empty()) {
    return all_names;
  }
  vector<string> output;
  output.reserve(get.projection_ids.size());
  for (auto proj_id : get.projection_ids) {
    if (proj_id >= all_names.size()) {
      throw InternalException("LogicalGet projection id is out of range");
    }
    output.push_back(all_names[proj_id]);
  }
  return output;
}

static string
StringifyPushedFiltersFromBindDataForDataFusion(const LogicalGet &get) {
  if (!get.bind_data) {
    return "";
  }

  auto *scan_bind = dynamic_cast<LanceScanBindData *>(get.bind_data.get());
  if (!scan_bind) {
    return "";
  }

  if (scan_bind->duckdb_pushed_filter_sql_parts.empty()) {
    return "";
  }

  string predicate;
  for (auto &part : scan_bind->duckdb_pushed_filter_sql_parts) {
    if (part.empty()) {
      continue;
    }
    if (predicate.empty()) {
      predicate = "(" + part + ")";
    } else {
      predicate += " AND (" + part + ")";
    }
  }
  return predicate;
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
                               LanceTableEntry &table_p, string dataset_uri_p,
                               string predicate_p, vector<string> set_columns_p,
                               vector<string> set_expressions_p,
                               idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), dataset_uri(std::move(dataset_uri_p)),
        predicate(std::move(predicate_p)),
        set_columns(std::move(set_columns_p)),
        set_expressions(std::move(set_expressions_p)) {}

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

  SourceResultType GetData(ExecutionContext &context, DataChunk &chunk,
                           OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<UpdateSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    if (set_columns.size() != set_expressions.size()) {
      throw InternalException("Lance UPDATE has mismatched SET columns/values");
    }

    auto open_path = dataset_uri;
    vector<string> option_keys;
    vector<string> option_values;
    ResolveLanceStorageOptions(context.client, dataset_uri, open_path,
                               option_keys, option_values);

    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                    value_ptrs);

    vector<const char *> set_col_ptrs;
    vector<const char *> set_expr_ptrs;
    set_col_ptrs.reserve(set_columns.size());
    set_expr_ptrs.reserve(set_expressions.size());
    for (idx_t i = 0; i < set_columns.size(); i++) {
      set_col_ptrs.push_back(set_columns[i].c_str());
      set_expr_ptrs.push_back(set_expressions[i].c_str());
    }

    void *txn = nullptr;
    uint64_t rows_updated = 0;
    auto rc = lance_overwrite_update_transaction_with_storage_options(
        open_path.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
        predicate.c_str(), set_col_ptrs.data(), set_expr_ptrs.data(),
        set_columns.size(), LANCE_DEFAULT_MAX_ROWS_PER_FILE,
        LANCE_DEFAULT_MAX_ROWS_PER_GROUP, LANCE_DEFAULT_MAX_BYTES_PER_FILE,
        &txn, &rows_updated);
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

    RegisterLancePendingAppend(context.client, table.ParentCatalog(),
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
    result["Predicate"] = predicate;
    result["SET Columns"] = StringUtil::Join(set_columns, "\n");
    result["SET Expressions"] = StringUtil::Join(set_expressions, "\n");
    return result;
  }

private:
  LanceTableEntry &table;
  string dataset_uri;
  string predicate;
  vector<string> set_columns;
  vector<string> set_expressions;
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

  vector<string> predicate_parts;
  auto table_filter_pred =
      StringifyTableFiltersForDataFusion(parts.get->table_filters, op.table);
  if (!table_filter_pred.empty()) {
    predicate_parts.push_back("(" + table_filter_pred + ")");
  }

  auto pushed_pred =
      StringifyPushedFiltersFromBindDataForDataFusion(*parts.get);
  if (!pushed_pred.empty()) {
    predicate_parts.push_back("(" + pushed_pred + ")");
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
      auto s =
          StringifyExprForDataFusion(*expr, *filter->children[0], op.table);
      predicate_parts.push_back("(" + s + ")");
    }
  }

  string predicate;
  for (auto &part : predicate_parts) {
    if (part.empty()) {
      continue;
    }
    if (predicate.empty()) {
      predicate = part;
    } else {
      predicate += " AND " + part;
    }
  }

  if (predicate.empty()) {
    throw NotImplementedException("Lance UPDATE requires a WHERE predicate "
                                  "(full-table UPDATE is not supported)");
  }

  vector<string> set_columns;
  vector<string> set_expressions;
  set_columns.reserve(op.columns.size());
  set_expressions.reserve(op.columns.size());

  if (!parts.projection) {
    throw InternalException("Lance UPDATE is missing projection");
  }

  if (parts.projection->children.empty() || !parts.projection->children[0]) {
    throw InternalException("Lance UPDATE projection has no child");
  }

  for (idx_t i = 0; i < op.columns.size(); i++) {
    auto &target_col = op.table.GetColumns().GetColumn(op.columns[i]);
    set_columns.push_back(target_col.Name());

    if (!op.expressions[i]) {
      throw InternalException("Lance UPDATE has null SET expression");
    }

    if (op.expressions[i]->GetExpressionType() ==
        ExpressionType::VALUE_DEFAULT) {
      throw NotImplementedException(
          "Lance UPDATE does not support SET column = DEFAULT");
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
    // Prefer DuckDB's expression stringification for SET expressions. This
    // keeps names stable across optimizer rewrites that can introduce
    // projections with NULL placeholders, while still producing
    // DataFusion-compatible SQL for simple identifiers and operators.
    auto expr_str = set_expr->ToString();
    set_expressions.push_back(std::move(expr_str));
  }

  return planner.Make<PhysicalLanceUpdateOverwrite>(
      op.types, *lance_table, lance_table->DatasetUri(), std::move(predicate),
      std::move(set_columns), std::move(set_expressions),
      op.estimated_cardinality);
}

} // namespace duckdb
