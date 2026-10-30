#include "lance_filter_ir.hpp"

#include "lance_common.hpp"
#include "lance_expr_ir.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/table_filter.hpp"

#include <cstdint>
#include <functional>

namespace duckdb {

bool LanceFilterIRSupportsLogicalType(const LogicalType &type) {
  return LanceExprIRSupportsLogicalType(type);
}

static constexpr char LANCE_FILTER_IR_MAGIC[] = {'L', 'F', 'T', '1'};

enum class LanceFilterIRTag : uint8_t {
  AND = 3,
  OR = 4,
  NOT = 5,
  IS_NULL = 7,
  IS_NOT_NULL = 8,
};

static constexpr uint8_t LANCE_FILTER_IR_REGEXP_MODE_PARTIAL_MATCH = 0;
static constexpr uint8_t LANCE_FILTER_IR_REGEXP_MODE_FULL_MATCH = 1;

static string LanceFilterIREncodeMessage(const string &root_node) {
  string out;
  out.append(LANCE_FILTER_IR_MAGIC, sizeof(LANCE_FILTER_IR_MAGIC));
  out.append(root_node);
  return out;
}

static bool TryEncodeLanceFilterIRColumnRef(const vector<string> &segments,
                                            string &out_ir) {
  return TryEncodeLanceExprIRColumnRef(segments, out_ir);
}

static bool TryEncodeLanceFilterIRColumnRef(const string &name,
                                            string &out_ir) {
  return TryEncodeLanceExprIRColumnRef(name, out_ir);
}

static bool TryEncodeLanceFilterIRLiteral(const Value &value, string &out_ir) {
  return TryEncodeLanceExprIRLiteral(value, out_ir);
}

static bool TryEncodeLanceFilterIRListLiteral(const Value &value,
                                              string &out_ir) {
  return TryEncodeLanceExprIRListLiteral(value, out_ir);
}

static bool TryEncodeLanceFilterIRComparisonOp(ExpressionType type,
                                               uint8_t &out_op) {
  return TryEncodeLanceExprIRComparisonOp(type, out_op);
}

static bool TryEncodeLanceFilterIRConjunction(LanceFilterIRTag tag,
                                              const vector<string> &children,
                                              string &out_ir) {
  if (tag == LanceFilterIRTag::AND) {
    return TryEncodeLanceExprIRAnd(children, out_ir);
  }
  if (tag == LanceFilterIRTag::OR) {
    return TryEncodeLanceExprIROr(children, out_ir);
  }
  return false;
}

static bool TryEncodeLanceFilterIRUnary(LanceFilterIRTag tag,
                                        const string &child, string &out_ir) {
  if (tag == LanceFilterIRTag::NOT) {
    return TryEncodeLanceExprIRNot(child, out_ir);
  }
  if (tag == LanceFilterIRTag::IS_NULL) {
    return TryEncodeLanceExprIRIsNull(child, out_ir);
  }
  if (tag == LanceFilterIRTag::IS_NOT_NULL) {
    return TryEncodeLanceExprIRIsNotNull(child, out_ir);
  }
  return false;
}

static bool TryEncodeLanceFilterIRComparison(uint8_t op, const string &left,
                                             const string &right,
                                             string &out_ir) {
  return TryEncodeLanceExprIRComparison(op, left, right, out_ir);
}

static bool TryEncodeLanceFilterIRInList(bool negated, const string &expr,
                                         const vector<string> &list,
                                         string &out_ir) {
  return TryEncodeLanceExprIRInList(negated, expr, list, out_ir);
}

static bool TryEncodeLanceFilterIRLike(bool case_insensitive, bool has_escape,
                                       uint8_t escape_char, const string &expr,
                                       const string &pattern, string &out_ir) {
  return TryEncodeLanceExprIRLike(case_insensitive, has_escape, escape_char,
                                  expr, pattern, out_ir);
}

static bool TryEncodeLanceFilterIRRegexp(uint8_t mode, bool has_flags,
                                         const string &expr,
                                         const string &pattern,
                                         const string &flags, string &out_ir) {
  return TryEncodeLanceExprIRRegexp(mode, has_flags, expr, pattern, flags,
                                    out_ir);
}

static bool TryEncodeLanceFilterIRScalarFunction(const string &name,
                                                 const vector<string> &args,
                                                 string &out_ir) {
  return TryEncodeLanceExprIRScalarFunction(name, args, out_ir);
}

static bool TryGetNonNullVarcharConstant(const Expression &expr,
                                         string &out_value) {
  if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
    auto &c = expr.Cast<BoundConstantExpression>();
    if (c.GetValue().IsNull()) {
      return false;
    }
    try {
      out_value =
          c.GetValue().DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
      return true;
    } catch (...) {
      return false;
    }
  }
  if (BoundCastExpression::IsCast(expr)) {
    auto &cast = expr.Cast<BoundFunctionExpression>();
    if (BoundCastExpression::IsTryCast(cast)) {
      return false;
    }
    if (!cast.GetChildren()[0] || cast.GetChildren()[0]->GetExpressionClass() !=
                                      ExpressionClass::BOUND_CONSTANT) {
      return false;
    }
    auto &c = cast.GetChildren()[0]->Cast<BoundConstantExpression>();
    if (c.GetValue().IsNull()) {
      return false;
    }
    try {
      auto casted = c.GetValue().DefaultCastAs(cast.GetReturnType());
      out_value = casted.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
      return true;
    } catch (...) {
      return false;
    }
  }
  return false;
}

static bool TryGetNonNullListConstant(const Expression &expr,
                                      Value &out_value) {
  if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
    auto &c = expr.Cast<BoundConstantExpression>();
    if (c.GetValue().IsNull() ||
        c.GetValue().type().id() != LogicalTypeId::LIST) {
      return false;
    }
    out_value = c.GetValue();
    return true;
  }
  if (BoundCastExpression::IsCast(expr)) {
    auto &cast = expr.Cast<BoundFunctionExpression>();
    if (BoundCastExpression::IsTryCast(cast)) {
      return false;
    }
    if (!cast.GetChildren()[0] || cast.GetChildren()[0]->GetExpressionClass() !=
                                      ExpressionClass::BOUND_CONSTANT) {
      return false;
    }
    auto &c = cast.GetChildren()[0]->Cast<BoundConstantExpression>();
    if (c.GetValue().IsNull()) {
      return false;
    }
    try {
      auto casted = c.GetValue().DefaultCastAs(cast.GetReturnType());
      if (casted.IsNull() || casted.type().id() != LogicalTypeId::LIST) {
        return false;
      }
      out_value = std::move(casted);
      return true;
    } catch (...) {
      return false;
    }
  }
  return false;
}

// DuckDB exposes the containment predicates under both a `list_*` and an
// `array_*` name; Lance's scalar-index planner matches DataFusion's
// `array_has_*` names, so the IR always carries the DataFusion spelling. The
// `&&` / `@>` / `<@` operator aliases are deliberately excluded.
static bool TryGetLanceContainmentFunctionName(const string &duckdb_name,
                                               string &out_name) {
  if (duckdb_name == "array_has_any" || duckdb_name == "list_has_any") {
    out_name = "array_has_any";
    return true;
  }
  if (duckdb_name == "array_has_all" || duckdb_name == "list_has_all") {
    out_name = "array_has_all";
    return true;
  }
  return false;
}

static bool TryBuildLanceTableFilterIRExpr(const vector<string> &column_refs,
                                           const TableFilter &filter,
                                           string &out_ir) {
  std::function<bool(const Expression &, string &)> build_from_expr;

  build_from_expr = [&](const Expression &expr, string &out_expr_ir) -> bool {
    if (BoundCastExpression::IsCast(expr)) {
      auto &cast = expr.Cast<BoundFunctionExpression>();
      if (BoundCastExpression::IsTryCast(cast)) {
        return false;
      }
      if (!cast.GetChildren()[0] ||
          cast.GetChildren()[0]->GetExpressionClass() !=
              ExpressionClass::BOUND_CONSTANT) {
        return false;
      }
      auto &c = cast.GetChildren()[0]->Cast<BoundConstantExpression>();
      try {
        auto casted = c.GetValue().DefaultCastAs(cast.GetReturnType());
        return TryEncodeLanceFilterIRLiteral(casted, out_expr_ir);
      } catch (...) {
        return false;
      }
    }
    if (BoundComparisonExpression::IsComparison(expr)) {
      auto &cmp = expr.Cast<BoundFunctionExpression>();
      uint8_t op = 0;
      if (!TryEncodeLanceFilterIRComparisonOp(cmp.GetExpressionType(), op)) {
        return false;
      }
      string lhs_ir, rhs_ir;
      if (!cmp.GetChildren()[0] || !cmp.GetChildren()[1]) {
        return false;
      }
      if (!build_from_expr(*cmp.GetChildren()[0], lhs_ir) ||
          !build_from_expr(*cmp.GetChildren()[1], rhs_ir)) {
        return false;
      }
      return TryEncodeLanceFilterIRComparison(op, lhs_ir, rhs_ir, out_expr_ir);
    }
    switch (expr.GetExpressionClass()) {
    case ExpressionClass::BOUND_REF: {
      auto index = expr.Cast<BoundReferenceExpression>().Index();
      if (index >= column_refs.size()) {
        return false;
      }
      out_expr_ir = column_refs[index];
      return true;
    }
    case ExpressionClass::BOUND_FUNCTION: {
      auto &func = expr.Cast<BoundFunctionExpression>();
      for (auto &child : func.GetChildren()) {
        if (!child) {
          return false;
        }
      }

      if (func.Function().GetName().GetIdentifierName() == "lower" ||
          func.Function().GetName().GetIdentifierName() == "upper") {
        if (func.GetChildren().size() != 1) {
          return false;
        }
        string input_ir;
        if (!build_from_expr(*func.GetChildren()[0], input_ir)) {
          return false;
        }
        vector<string> args;
        args.push_back(std::move(input_ir));
        return TryEncodeLanceFilterIRScalarFunction(
            func.Function().GetName().GetIdentifierName(), args, out_expr_ir);
      }

      if (func.Function().GetName().GetIdentifierName() == "starts_with" ||
          func.Function().GetName().GetIdentifierName() == "ends_with" ||
          func.Function().GetName().GetIdentifierName() == "contains") {
        if (func.GetChildren().size() != 2) {
          return false;
        }

        string input_ir;
        if (!build_from_expr(*func.GetChildren()[0], input_ir)) {
          return false;
        }

        string needle_value;
        if (!TryGetNonNullVarcharConstant(*func.GetChildren()[1],
                                          needle_value)) {
          return false;
        }
        string needle_ir;
        if (!TryEncodeLanceFilterIRLiteral(Value(needle_value), needle_ir)) {
          return false;
        }

        vector<string> args;
        args.push_back(std::move(input_ir));
        args.push_back(std::move(needle_ir));
        return TryEncodeLanceFilterIRScalarFunction(
            func.Function().GetName().GetIdentifierName(), args, out_expr_ir);
      }

      string containment_name;
      if (TryGetLanceContainmentFunctionName(
              func.Function().GetName().GetIdentifierName(),
              containment_name)) {
        if (func.GetChildren().size() != 2) {
          return false;
        }

        string input_ir;
        if (!build_from_expr(*func.GetChildren()[0], input_ir)) {
          return false;
        }

        Value needle_value;
        if (!TryGetNonNullListConstant(*func.GetChildren()[1], needle_value)) {
          return false;
        }
        string needle_ir;
        if (!TryEncodeLanceFilterIRListLiteral(needle_value, needle_ir)) {
          return false;
        }

        vector<string> args;
        args.push_back(std::move(input_ir));
        args.push_back(std::move(needle_ir));
        return TryEncodeLanceFilterIRScalarFunction(containment_name, args,
                                                    out_expr_ir);
      }

      return false;
    }
    case ExpressionClass::BOUND_CONSTANT: {
      auto &c = expr.Cast<BoundConstantExpression>();
      return TryEncodeLanceFilterIRLiteral(c.GetValue(), out_expr_ir);
    }

    case ExpressionClass::BOUND_CONJUNCTION: {
      auto &conj = expr.Cast<BoundConjunctionExpression>();
      LanceFilterIRTag tag;
      if (conj.GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
        tag = LanceFilterIRTag::AND;
      } else if (conj.GetExpressionType() == ExpressionType::CONJUNCTION_OR) {
        tag = LanceFilterIRTag::OR;
      } else {
        return false;
      }
      vector<string> children;
      children.reserve(conj.GetChildren().size());
      for (auto &child : conj.GetChildren()) {
        if (!child) {
          return false;
        }
        string child_ir;
        if (!build_from_expr(*child, child_ir)) {
          return false;
        }
        children.push_back(std::move(child_ir));
      }
      return TryEncodeLanceFilterIRConjunction(tag, children, out_expr_ir);
    }
    case ExpressionClass::BOUND_OPERATOR: {
      auto &op = expr.Cast<BoundOperatorExpression>();
      if (op.GetExpressionType() == ExpressionType::OPERATOR_NOT) {
        if (op.GetChildren().size() != 1 || !op.GetChildren()[0]) {
          return false;
        }
        string child_ir;
        if (!build_from_expr(*op.GetChildren()[0], child_ir)) {
          return false;
        }
        return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::NOT, child_ir,
                                           out_expr_ir);
      }
      if (op.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ||
          op.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
        if (op.GetChildren().size() != 1 || !op.GetChildren()[0]) {
          return false;
        }
        string child_ir;
        if (!build_from_expr(*op.GetChildren()[0], child_ir)) {
          return false;
        }
        return TryEncodeLanceFilterIRUnary(
            op.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL
                ? LanceFilterIRTag::IS_NULL
                : LanceFilterIRTag::IS_NOT_NULL,
            child_ir, out_expr_ir);
      }
      if (op.GetExpressionType() == ExpressionType::COMPARE_IN ||
          op.GetExpressionType() == ExpressionType::COMPARE_NOT_IN) {
        if (op.GetChildren().size() < 2 || !op.GetChildren()[0]) {
          return false;
        }
        string lhs_ir;
        if (!build_from_expr(*op.GetChildren()[0], lhs_ir)) {
          return false;
        }
        vector<string> values;
        values.reserve(op.GetChildren().size() - 1);
        for (idx_t i = 1; i < op.GetChildren().size(); i++) {
          if (!op.GetChildren()[i] ||
              op.GetChildren()[i]->GetExpressionClass() !=
                  ExpressionClass::BOUND_CONSTANT) {
            return false;
          }
          auto &c = op.GetChildren()[i]->Cast<BoundConstantExpression>();
          string lit_ir;
          if (!TryEncodeLanceFilterIRLiteral(c.GetValue(), lit_ir)) {
            return false;
          }
          values.push_back(std::move(lit_ir));
        }
        return TryEncodeLanceFilterIRInList(op.GetExpressionType() ==
                                                ExpressionType::COMPARE_NOT_IN,
                                            lhs_ir, values, out_expr_ir);
      }
      return false;
    }
    default:
      return false;
    }
  };

  const auto &expression_filter =
      ExpressionFilter::GetExpressionFilter(filter, "Lance filter pushdown");
  return expression_filter.expr &&
         build_from_expr(*expression_filter.expr, out_ir);
}

LanceFilterIRBuildResult
ProbeLanceTableFilterIR(LogicalGet &get, const vector<string> &names,
                        const vector<LogicalType> &types) {
  vector<idx_t> projection_ids;
  TableFunctionInitInput probe_input(get.bind_data.get(), get.GetColumnIds(),
                                     projection_ids, &get.table_filters);
  return BuildLanceTableFilterIRParts(names, types, probe_input, false);
}

LanceFilterIRBuildResult BuildLanceTableFilterIRParts(
    const vector<string> &names, const vector<LogicalType> &types,
    const TableFunctionInitInput &input, bool exclude_computed_columns) {
  LanceFilterIRBuildResult result;
  if (!input.filters) {
    return result;
  }

  result.parts.reserve(input.filters->FilterCount());

  for (auto &it : *input.filters) {
    auto scan_col_idx = it.GetIndex();
    auto &filter = it.Filter();

    if (scan_col_idx >= input.column_ids.size()) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }
    auto col_id = input.column_ids[scan_col_idx];
    if (col_id >= names.size() || col_id >= types.size()) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }
    if (exclude_computed_columns && IsComputedSearchColumn(names[col_id])) {
      result.all_filters_pushed = false;
      continue;
    }
    if (!LanceFilterIRSupportsLogicalType(types[col_id])) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }

    string col_ref_ir;
    if (!TryEncodeLanceFilterIRColumnRef(names[col_id], col_ref_ir)) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }
    string filter_ir;
    if (!TryBuildLanceTableFilterIRExpr({col_ref_ir}, filter, filter_ir)) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }

    result.parts.push_back(std::move(filter_ir));
  }

  for (const auto &filter : input.filters->GetMultiColumnFilters()) {
    const auto &expression_filter = ExpressionFilter::GetExpressionFilter(
        *filter, "Lance multi-column pushdown");
    vector<string> column_refs;
    bool supported = true;
    bool computed = false;
    for (auto projection : expression_filter.column_indexes) {
      if (projection >= input.column_ids.size()) {
        supported = false;
        break;
      }
      auto column_id = input.column_ids[projection];
      if (column_id >= names.size() || column_id >= types.size()) {
        supported = false;
        break;
      }
      if (exclude_computed_columns &&
          IsComputedSearchColumn(names[column_id])) {
        computed = true;
        supported = false;
        break;
      }
      string column_ref;
      if (!LanceFilterIRSupportsLogicalType(types[column_id]) ||
          !TryEncodeLanceFilterIRColumnRef(names[column_id], column_ref)) {
        supported = false;
        break;
      }
      column_refs.push_back(std::move(column_ref));
    }
    string filter_ir;
    if (!supported ||
        !TryBuildLanceTableFilterIRExpr(column_refs, *filter, filter_ir)) {
      result.all_filters_pushed = false;
      if (!computed) {
        result.all_prefilterable_filters_pushed = false;
      }
      continue;
    }
    result.parts.push_back(std::move(filter_ir));
  }

  return result;
}

static bool TryBuildLanceExprColumnRefIR(const LogicalGet &get,
                                         const vector<string> &names,
                                         const vector<LogicalType> &types,
                                         bool exclude_computed_columns,
                                         const Expression &expr,
                                         string &out_ir) {
  vector<string> segments;
  LogicalType leaf_type;

  std::function<bool(const Expression &, vector<string> &, LogicalType &)>
      build_segments;

  build_segments = [&](const Expression &node, vector<string> &out_segments,
                       LogicalType &out_leaf_type) -> bool {
    if (node.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
      auto &colref = node.Cast<BoundColumnRefExpression>();
      if (colref.Depth() != 0) {
        return false;
      }
      if (colref.Binding().table_index != get.table_index) {
        return false;
      }
      auto &column_ids = get.GetColumnIds();
      if (colref.Binding().column_index >= column_ids.size()) {
        return false;
      }
      auto &col_index = column_ids[colref.Binding().column_index];
      if (col_index.IsVirtualColumn()) {
        return false;
      }
      auto col_id = col_index.GetPrimaryIndex();
      if (col_id >= names.size() || col_id >= types.size()) {
        return false;
      }

      out_segments.clear();
      out_segments.push_back(names[col_id]);
      out_leaf_type = types[col_id];

      vector<idx_t> child_path;
      child_path.reserve(col_index.GetChildIndexes().size());

      std::function<bool(const vector<ColumnIndex> &)> linearize_child_indexes;
      linearize_child_indexes =
          [&](const vector<ColumnIndex> &children) -> bool {
        if (children.empty()) {
          return true;
        }
        if (children.size() > 1) {
          for (auto &child : children) {
            if (!child.GetChildIndexes().empty()) {
              return false;
            }
            child_path.push_back(child.GetPrimaryIndex());
          }
          return true;
        }
        auto &child = children[0];
        child_path.push_back(child.GetPrimaryIndex());
        return linearize_child_indexes(child.GetChildIndexes());
      };

      if (!linearize_child_indexes(col_index.GetChildIndexes())) {
        return false;
      }

      for (auto child_idx : child_path) {
        if (out_leaf_type.id() != LogicalTypeId::STRUCT) {
          return false;
        }
        if (child_idx >= StructType::GetChildCount(out_leaf_type)) {
          return false;
        }
        out_segments.push_back(
            StructType::GetChildName(out_leaf_type, child_idx)
                .GetIdentifierName());
        out_leaf_type = StructType::GetChildType(out_leaf_type, child_idx);
      }
      return true;
    }

    if (node.GetExpressionClass() == ExpressionClass::BOUND_REF) {
      auto &ref = node.Cast<BoundReferenceExpression>();
      auto &column_ids = get.GetColumnIds();
      if (ref.Index() >= column_ids.size()) {
        return false;
      }
      auto &col_index = column_ids[ref.Index()];
      if (col_index.IsVirtualColumn()) {
        return false;
      }
      auto col_id = col_index.GetPrimaryIndex();
      if (col_id >= names.size() || col_id >= types.size()) {
        return false;
      }

      out_segments.clear();
      out_segments.push_back(names[col_id]);
      out_leaf_type = types[col_id];

      vector<idx_t> child_path;
      child_path.reserve(col_index.GetChildIndexes().size());

      std::function<bool(const vector<ColumnIndex> &)> linearize_child_indexes;
      linearize_child_indexes =
          [&](const vector<ColumnIndex> &children) -> bool {
        if (children.empty()) {
          return true;
        }
        if (children.size() > 1) {
          for (auto &child : children) {
            if (!child.GetChildIndexes().empty()) {
              return false;
            }
            child_path.push_back(child.GetPrimaryIndex());
          }
          return true;
        }
        auto &child = children[0];
        child_path.push_back(child.GetPrimaryIndex());
        return linearize_child_indexes(child.GetChildIndexes());
      };

      if (!linearize_child_indexes(col_index.GetChildIndexes())) {
        return false;
      }

      for (auto child_idx : child_path) {
        if (out_leaf_type.id() != LogicalTypeId::STRUCT) {
          return false;
        }
        if (child_idx >= StructType::GetChildCount(out_leaf_type)) {
          return false;
        }
        out_segments.push_back(
            StructType::GetChildName(out_leaf_type, child_idx)
                .GetIdentifierName());
        out_leaf_type = StructType::GetChildType(out_leaf_type, child_idx);
      }
      return true;
    }

    if (node.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
      auto &func = node.Cast<BoundFunctionExpression>();
      if (func.Function().GetName().GetIdentifierName() != "struct_extract" &&
          func.Function().GetName().GetIdentifierName() !=
              "struct_extract_at") {
        return false;
      }
      if (func.GetChildren().size() != 2 || !func.GetChildren()[0] ||
          !func.GetChildren()[1]) {
        return false;
      }

      vector<string> base_segments;
      LogicalType base_type;
      if (!build_segments(*func.GetChildren()[0], base_segments, base_type)) {
        return false;
      }
      if (base_type.id() != LogicalTypeId::STRUCT) {
        return false;
      }

      if (func.GetChildren()[1]->GetExpressionClass() !=
          ExpressionClass::BOUND_CONSTANT) {
        return false;
      }
      auto &c = func.GetChildren()[1]->Cast<BoundConstantExpression>();
      if (c.GetValue().IsNull()) {
        return false;
      }

      idx_t child_idx = 0;
      string child_name;
      if (func.Function().GetName().GetIdentifierName() == "struct_extract") {
        if (c.GetValue().type().id() != LogicalTypeId::VARCHAR) {
          return false;
        }
        child_name = c.GetValue().GetValue<string>();
        bool found = false;
        for (idx_t i = 0; i < StructType::GetChildCount(base_type); i++) {
          if (StructType::GetChildName(base_type, i).GetIdentifierName() ==
              child_name) {
            child_idx = i;
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      } else {
        try {
          auto one_based = c.GetValue()
                               .DefaultCastAs(LogicalType::BIGINT)
                               .GetValue<int64_t>();
          if (one_based <= 0) {
            return false;
          }
          child_idx = static_cast<idx_t>(one_based - 1);
        } catch (...) {
          return false;
        }
        if (child_idx >= StructType::GetChildCount(base_type)) {
          return false;
        }
        child_name =
            StructType::GetChildName(base_type, child_idx).GetIdentifierName();
        if (child_name.empty()) {
          // Fallback: for unnamed structs, the name can be empty; encode a
          // stable segment anyway by using the 1-based index.
          child_name = to_string(child_idx + 1);
        }
      }

      base_segments.push_back(child_name);
      out_segments = std::move(base_segments);
      out_leaf_type = StructType::GetChildType(base_type, child_idx);
      return true;
    }

    return false;
  };

  if (!build_segments(expr, segments, leaf_type)) {
    return false;
  }
  if (segments.empty()) {
    return false;
  }
  if (exclude_computed_columns && IsComputedSearchColumn(segments[0])) {
    return false;
  }
  if (!LanceFilterIRSupportsLogicalType(leaf_type)) {
    return false;
  }
  return TryEncodeLanceFilterIRColumnRef(segments, out_ir);
}

bool TryBuildLanceExprFilterIR(const LogicalGet &get,
                               const vector<string> &names,
                               const vector<LogicalType> &types,
                               bool exclude_computed_columns,
                               const Expression &expr, string &out_ir) {
  if (BoundCastExpression::IsCast(expr)) {
    auto &cast = expr.Cast<BoundFunctionExpression>();
    if (BoundCastExpression::IsTryCast(cast)) {
      return false;
    }
    if (cast.GetChildren()[0]->GetExpressionClass() !=
        ExpressionClass::BOUND_CONSTANT) {
      return false;
    }
    auto &c = cast.GetChildren()[0]->Cast<BoundConstantExpression>();
    try {
      auto casted = c.GetValue().DefaultCastAs(cast.GetReturnType());
      return TryEncodeLanceFilterIRLiteral(casted, out_ir);
    } catch (...) {
      return false;
    }
  }
  if (BoundComparisonExpression::IsComparison(expr)) {
    auto &cmp = expr.Cast<BoundFunctionExpression>();
    uint8_t op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(cmp.GetExpressionType(), op)) {
      return false;
    }
    string lhs_ir, rhs_ir;
    if (!TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *cmp.GetChildren()[0], lhs_ir) ||
        !TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *cmp.GetChildren()[1], rhs_ir)) {
      return false;
    }
    return TryEncodeLanceFilterIRComparison(op, lhs_ir, rhs_ir, out_ir);
  }
  if (expr.GetExpressionType() == ExpressionType::COMPARE_BETWEEN) {
    auto &between = expr.Cast<BoundFunctionExpression>();
    uint8_t lower_op = 0;
    uint8_t upper_op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(
            BoundBetweenExpression::LowerComparisonType(between), lower_op) ||
        !TryEncodeLanceFilterIRComparisonOp(
            BoundBetweenExpression::UpperComparisonType(between), upper_op)) {
      return false;
    }
    string input_ir, lower_ir, upper_ir;
    if (!TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *between.GetChildren()[0], input_ir) ||
        !TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *between.GetChildren()[1], lower_ir) ||
        !TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *between.GetChildren()[2], upper_ir)) {
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
  switch (expr.GetExpressionClass()) {
  case ExpressionClass::BOUND_COLUMN_REF:
  case ExpressionClass::BOUND_REF:
    return TryBuildLanceExprColumnRefIR(get, names, types,
                                        exclude_computed_columns, expr, out_ir);
  case ExpressionClass::BOUND_FUNCTION: {
    auto &func = expr.Cast<BoundFunctionExpression>();
    if (func.Function().GetName().GetIdentifierName() == "struct_extract" ||
        func.Function().GetName().GetIdentifierName() == "struct_extract_at") {
      return TryBuildLanceExprColumnRefIR(
          get, names, types, exclude_computed_columns, expr, out_ir);
    }

    if (func.Function().GetName().GetIdentifierName() == "lower" ||
        func.Function().GetName().GetIdentifierName() == "upper") {
      if (func.GetChildren().size() != 1 || !func.GetChildren()[0]) {
        return false;
      }
      string input_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns,
                                     *func.GetChildren()[0], input_ir)) {
        return false;
      }
      vector<string> args;
      args.push_back(std::move(input_ir));
      return TryEncodeLanceFilterIRScalarFunction(
          func.Function().GetName().GetIdentifierName(), args, out_ir);
    }

    if (func.Function().GetName().GetIdentifierName() == "starts_with" ||
        func.Function().GetName().GetIdentifierName() == "ends_with" ||
        func.Function().GetName().GetIdentifierName() == "contains") {
      if (func.GetChildren().size() != 2 || !func.GetChildren()[0] ||
          !func.GetChildren()[1]) {
        return false;
      }
      string input_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns,
                                     *func.GetChildren()[0], input_ir)) {
        return false;
      }

      string needle_value;
      if (!TryGetNonNullVarcharConstant(*func.GetChildren()[1], needle_value)) {
        return false;
      }
      string needle_ir;
      if (!TryEncodeLanceFilterIRLiteral(Value(needle_value), needle_ir)) {
        return false;
      }

      vector<string> args;
      args.push_back(std::move(input_ir));
      args.push_back(std::move(needle_ir));
      return TryEncodeLanceFilterIRScalarFunction(
          func.Function().GetName().GetIdentifierName(), args, out_ir);
    }

    string containment_name;
    if (TryGetLanceContainmentFunctionName(
            func.Function().GetName().GetIdentifierName(), containment_name)) {
      if (func.GetChildren().size() != 2 || !func.GetChildren()[0] ||
          !func.GetChildren()[1]) {
        return false;
      }
      string input_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns,
                                     *func.GetChildren()[0], input_ir)) {
        return false;
      }

      Value needle_value;
      if (!TryGetNonNullListConstant(*func.GetChildren()[1], needle_value)) {
        return false;
      }
      string needle_ir;
      if (!TryEncodeLanceFilterIRListLiteral(needle_value, needle_ir)) {
        return false;
      }

      vector<string> args;
      args.push_back(std::move(input_ir));
      args.push_back(std::move(needle_ir));
      return TryEncodeLanceFilterIRScalarFunction(containment_name, args,
                                                  out_ir);
    }

    if (func.Function().GetName().GetIdentifierName() == "regexp_matches" ||
        func.Function().GetName().GetIdentifierName() == "regexp_full_match") {
      if (func.GetChildren().size() != 2 && func.GetChildren().size() != 3) {
        return false;
      }
      for (auto &child : func.GetChildren()) {
        if (!child) {
          return false;
        }
      }

      string input_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns,
                                     *func.GetChildren()[0], input_ir)) {
        return false;
      }

      string pattern_value;
      if (!TryGetNonNullVarcharConstant(*func.GetChildren()[1],
                                        pattern_value)) {
        return false;
      }
      string pattern_ir;
      if (!TryEncodeLanceFilterIRLiteral(Value(pattern_value), pattern_ir)) {
        return false;
      }

      bool has_flags = func.GetChildren().size() == 3;
      string flags_ir;
      if (has_flags) {
        string flags_value;
        if (!TryGetNonNullVarcharConstant(*func.GetChildren()[2],
                                          flags_value)) {
          return false;
        }
        if (!TryEncodeLanceFilterIRLiteral(Value(flags_value), flags_ir)) {
          return false;
        }
      }

      uint8_t mode =
          func.Function().GetName().GetIdentifierName() == "regexp_full_match"
              ? LANCE_FILTER_IR_REGEXP_MODE_FULL_MATCH
              : LANCE_FILTER_IR_REGEXP_MODE_PARTIAL_MATCH;
      return TryEncodeLanceFilterIRRegexp(mode, has_flags, input_ir, pattern_ir,
                                          flags_ir, out_ir);
    }

    bool case_insensitive = false;
    bool has_escape = false;
    if (func.Function().GetName().GetIdentifierName() == "~~") {
      case_insensitive = false;
      has_escape = false;
    } else if (func.Function().GetName().GetIdentifierName() == "~~*") {
      case_insensitive = true;
      has_escape = false;
    } else if (func.Function().GetName().GetIdentifierName() == "like_escape") {
      case_insensitive = false;
      has_escape = true;
    } else if (func.Function().GetName().GetIdentifierName() ==
               "ilike_escape") {
      case_insensitive = true;
      has_escape = true;
    } else {
      return false;
    }

    if ((has_escape && func.GetChildren().size() != 3) ||
        (!has_escape && func.GetChildren().size() != 2)) {
      return false;
    }
    for (auto &child : func.GetChildren()) {
      if (!child) {
        return false;
      }
    }

    string input_ir;
    if (!TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *func.GetChildren()[0], input_ir)) {
      return false;
    }

    string pattern_value;
    if (!TryGetNonNullVarcharConstant(*func.GetChildren()[1], pattern_value)) {
      return false;
    }
    string pattern_ir;
    if (!TryEncodeLanceFilterIRLiteral(Value(pattern_value), pattern_ir)) {
      return false;
    }

    uint8_t escape_char = 0;
    if (has_escape) {
      string escape_value;
      if (!TryGetNonNullVarcharConstant(*func.GetChildren()[2], escape_value)) {
        return false;
      }
      if (escape_value.size() != 1) {
        return false;
      }
      escape_char = static_cast<uint8_t>(escape_value[0]);
      if (escape_char != static_cast<uint8_t>('\\')) {
        return false;
      }
    }

    return TryEncodeLanceFilterIRLike(case_insensitive, has_escape, escape_char,
                                      input_ir, pattern_ir, out_ir);
  }
  case ExpressionClass::BOUND_CONSTANT: {
    auto &c = expr.Cast<BoundConstantExpression>();
    return TryEncodeLanceFilterIRLiteral(c.GetValue(), out_ir);
  }

  case ExpressionClass::BOUND_CONJUNCTION: {
    auto &conj = expr.Cast<BoundConjunctionExpression>();
    LanceFilterIRTag tag;
    if (conj.GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
      tag = LanceFilterIRTag::AND;
    } else if (conj.GetExpressionType() == ExpressionType::CONJUNCTION_OR) {
      tag = LanceFilterIRTag::OR;
    } else {
      return false;
    }
    vector<string> children;
    children.reserve(conj.GetChildren().size());
    for (auto &child : conj.GetChildren()) {
      string child_ir;
      if (!TryBuildLanceExprFilterIR(
              get, names, types, exclude_computed_columns, *child, child_ir)) {
        return false;
      }
      children.push_back(std::move(child_ir));
    }
    return TryEncodeLanceFilterIRConjunction(tag, children, out_ir);
  }
  case ExpressionClass::BOUND_OPERATOR: {
    auto &op = expr.Cast<BoundOperatorExpression>();
    if (op.GetExpressionType() == ExpressionType::OPERATOR_NOT) {
      if (op.GetChildren().size() != 1) {
        return false;
      }
      string child_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns,
                                     *op.GetChildren()[0], child_ir)) {
        return false;
      }
      return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::NOT, child_ir,
                                         out_ir);
    }
    if (op.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ||
        op.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
      if (op.GetChildren().size() != 1) {
        return false;
      }
      string child_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns,
                                     *op.GetChildren()[0], child_ir)) {
        return false;
      }
      return TryEncodeLanceFilterIRUnary(
          op.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL
              ? LanceFilterIRTag::IS_NULL
              : LanceFilterIRTag::IS_NOT_NULL,
          child_ir, out_ir);
    }
    if (op.GetExpressionType() == ExpressionType::COMPARE_IN ||
        op.GetExpressionType() == ExpressionType::COMPARE_NOT_IN) {
      if (op.GetChildren().size() < 2) {
        return false;
      }
      string lhs_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns,
                                     *op.GetChildren()[0], lhs_ir)) {
        return false;
      }
      vector<string> values;
      values.reserve(op.GetChildren().size() - 1);
      for (idx_t i = 1; i < op.GetChildren().size(); i++) {
        if (op.GetChildren()[i]->GetExpressionClass() !=
            ExpressionClass::BOUND_CONSTANT) {
          return false;
        }
        auto &c = op.GetChildren()[i]->Cast<BoundConstantExpression>();
        string lit_ir;
        if (!TryEncodeLanceFilterIRLiteral(c.GetValue(), lit_ir)) {
          return false;
        }
        values.push_back(std::move(lit_ir));
      }
      return TryEncodeLanceFilterIRInList(op.GetExpressionType() ==
                                              ExpressionType::COMPARE_NOT_IN,
                                          lhs_ir, values, out_ir);
    }
    return false;
  }

  default:
    return false;
  }
}

bool TryEncodeLanceFilterIRMessage(const vector<string> &parts,
                                   string &out_msg) {
  out_msg.clear();
  if (parts.empty()) {
    return true;
  }
  string root_node;
  if (parts.size() == 1) {
    root_node = parts[0];
  } else if (!TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND, parts,
                                                root_node)) {
    return false;
  }
  out_msg = LanceFilterIREncodeMessage(root_node);
  return true;
}

} // namespace duckdb
