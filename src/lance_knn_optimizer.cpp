#include "lance_knn_optimizer.hpp"

#include "duckdb.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_filter_ir.hpp"
#include "lance_knn_scan.hpp"
#include "lance_scan_bind_data.hpp"

#include <cstring>
#include <unordered_set>

namespace duckdb {

namespace {

// Hard ceiling — even if the user sets lance_knn_max_k higher, we refuse to
// rewrite past this. Keeps a typo'd setting from pinning many GB of state
// inside Lance.
constexpr idx_t LANCE_KNN_ABSOLUTE_MAX_K = 1000000;

constexpr const char *LANCE_KNN_NPROBES_SETTING = "lance_knn_nprobes";
constexpr const char *LANCE_KNN_REFINE_FACTOR_SETTING =
    "lance_knn_refine_factor";
constexpr const char *LANCE_KNN_EF_SEARCH_SETTING = "lance_knn_ef_search";
constexpr const char *LANCE_KNN_PREFILTER_SETTING = "lance_knn_prefilter";
constexpr const char *LANCE_KNN_USE_INDEX_SETTING = "lance_knn_use_index";
constexpr const char *LANCE_KNN_MAX_K_SETTING = "lance_knn_max_k";
constexpr uint64_t LANCE_KNN_MAX_K_DEFAULT = 10000;

// Mirrors the helper in lance_scan.cpp — kept local to avoid coupling the
// optimizer header to the scan internals.
bool IsLanceScanTableFunction(const TableFunction &fn) {
  return fn.name == "__lance_scan" || fn.name == "__lance_table_scan" ||
         fn.name == "__lance_namespace_scan";
}

uint64_t GetKNNOptionUInt64(ClientContext &context, const char *name,
                            uint64_t fallback) {
  Value val;
  if (!context.TryGetCurrentSetting(name, val) || val.IsNull()) {
    return fallback;
  }
  auto v = val.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  if (v <= 0) {
    return 0;
  }
  return NumericCast<uint64_t>(v);
}

bool GetKNNOptionBool(ClientContext &context, const char *name, bool fallback) {
  Value val;
  if (!context.TryGetCurrentSetting(name, val) || val.IsNull()) {
    return fallback;
  }
  return val.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
}

// Maps the DuckDB distance function name to the metric tag Lance uses.
// Returns false for any unsupported / unrecognised distance function so the
// caller can fall back to the original plan.
//
// All three supported scalar distances are symmetric in the value they
// produce — L2 and cosine are obviously so, and negative inner product
// satisfies -(a·b) = -(b·a). The caller can therefore accept either
// argument order without altering semantics.
bool DistanceFunctionToMetric(const string &fn_name, string &out_metric) {
  if (fn_name == "array_distance" || fn_name == "list_distance") {
    out_metric = "l2";
    return true;
  }
  if (fn_name == "array_cosine_distance" || fn_name == "list_cosine_distance") {
    out_metric = "cosine";
    return true;
  }
  if (fn_name == "array_negative_inner_product" ||
      fn_name == "list_negative_inner_product") {
    out_metric = "dot";
    return true;
  }
  return false;
}

// Extracts query floats from a BoundConstantExpression. Accepts both
// FLOAT[N] (ARRAY) and FLOAT[] (LIST) constants — DuckDB binds the literal
// to one or the other depending on the distance function's signature
// (array_distance prefers ARRAY, list_distance prefers LIST). Returns false
// if the constant is not a float sequence, since the Lance KNN FFI takes a
// `const float *` and we are not willing to silently insert a cast.
bool TryExtractQueryVector(const Expression &expr, vector<float> &out) {
  if (expr.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
    return false;
  }
  auto &constant = expr.Cast<BoundConstantExpression>();
  if (constant.value.IsNull()) {
    return false;
  }
  auto &type = constant.value.type();
  const vector<Value> *children_ptr = nullptr;
  if (type.id() == LogicalTypeId::ARRAY) {
    if (ArrayType::GetChildType(type).id() != LogicalTypeId::FLOAT) {
      return false;
    }
    children_ptr = &ArrayValue::GetChildren(constant.value);
  } else if (type.id() == LogicalTypeId::LIST) {
    if (ListType::GetChildType(type).id() != LogicalTypeId::FLOAT) {
      return false;
    }
    children_ptr = &ListValue::GetChildren(constant.value);
  } else {
    return false;
  }
  out.clear();
  out.reserve(children_ptr->size());
  for (auto &child : *children_ptr) {
    if (child.IsNull()) {
      return false;
    }
    out.push_back(child.GetValue<float>());
  }
  return true;
}

// Walks past any BOUND_CAST nodes and returns the underlying expression.
// Cast chains are common for ORDER BY expressions (DuckDB inserts an explicit
// cast to align with the projection output type).
const Expression *PeelCasts(const Expression *expr) {
  while (expr && expr->GetExpressionClass() == ExpressionClass::BOUND_CAST) {
    auto &cast = expr->Cast<BoundCastExpression>();
    expr = cast.child.get();
  }
  return expr;
}

// Resolve a column ref pointing at the projection to the underlying
// expression in the projection's select list, peeling casts as we go. If the
// column ref does not bind to `proj.table_index` we leave `expr` untouched.
const Expression *ResolveThroughProjection(const Expression *expr,
                                           const LogicalProjection *proj) {
  expr = PeelCasts(expr);
  if (!expr || !proj) {
    return expr;
  }
  if (expr->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
    return expr;
  }
  auto &ref = expr->Cast<BoundColumnRefExpression>();
  if (ref.binding.table_index != proj->table_index) {
    return expr;
  }
  if (ref.binding.column_index >= proj->expressions.size()) {
    return expr;
  }
  return PeelCasts(proj->expressions[ref.binding.column_index].get());
}

bool TryResolveVectorColumn(const Expression &expr, const LogicalGet &get,
                            string &out_name) {
  if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
    return false;
  }
  auto &ref = expr.Cast<BoundColumnRefExpression>();
  if (ref.binding.table_index != get.table_index) {
    return false;
  }
  auto &col_ids = get.GetColumnIds();
  if (ref.binding.column_index >= col_ids.size()) {
    return false;
  }
  auto col_id = col_ids[ref.binding.column_index].GetPrimaryIndex();
  if (col_id >= get.names.size()) {
    return false;
  }
  if (ref.return_type.id() != LogicalTypeId::ARRAY) {
    return false;
  }
  // FLOAT[N] (Float32) is the only element type currently exercised end-to-
  // end through the KNN FFI: lance_create_knn_stream_ir takes a const
  // float * query and Lance's KNN executor expects the column to match.
  //
  // Other Arrow array element types fall through to a silent fallback:
  //   - DOUBLE[N] (Float64): Lance can store this, but routing it through
  //     the Float32 FFI requires a query-vector cast (loses precision) or
  //     a new Float64 FFI variant. Until that's wired we leave such
  //     columns on the DuckDB scan path — correct, just unindexed.
  //   - HALF_FLOAT[N] (Float16): LanceCoerceArrowSchemaForDuckDB converts
  //     these to FLOAT[N] before DuckDB sees them, so a DuckDB-visible
  //     HALF_FLOAT array shouldn't reach this point.
  //   - INTEGER / other numeric element types are not vector embeddings.
  //
  // If you change the supported types here, also update the rebind path
  // in lance_knn_scan.cpp and add a dataset fixture + test.
  auto &child_type = ArrayType::GetChildType(ref.return_type);
  if (child_type.id() != LogicalTypeId::FLOAT) {
    return false;
  }
  out_name = get.names[col_id];
  return true;
}

// Inspects the distance function call, finds the (column, query) sides, and
// returns the metric tag. Returns false if the call shape does not satisfy
// the rewrite preconditions.
struct DistanceCallInfo {
  string metric;
  string vector_column;
  vector<float> query;
};

bool TryDescribeDistanceCall(const Expression &expr, const LogicalGet &get,
                             DistanceCallInfo &out) {
  if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
    return false;
  }
  auto &func = expr.Cast<BoundFunctionExpression>();
  if (func.children.size() != 2) {
    return false;
  }
  string metric;
  if (!DistanceFunctionToMetric(func.function.name, metric)) {
    return false;
  }

  auto *lhs = PeelCasts(func.children[0].get());
  auto *rhs = PeelCasts(func.children[1].get());
  if (!lhs || !rhs) {
    return false;
  }

  vector<float> query;
  string vector_column;
  if (TryExtractQueryVector(*rhs, query) &&
      TryResolveVectorColumn(*lhs, get, vector_column)) {
    out.metric = std::move(metric);
    out.vector_column = std::move(vector_column);
    out.query = std::move(query);
    return true;
  }
  if (TryExtractQueryVector(*lhs, query) &&
      TryResolveVectorColumn(*rhs, get, vector_column)) {
    out.metric = std::move(metric);
    out.vector_column = std::move(vector_column);
    out.query = std::move(query);
    return true;
  }
  return false;
}

// Wraps the metric-lookup FFI via the dataset cache entry. Returns true when
// a vector index exists *and* its metric matches the requested one. On any
// error or mismatch we set `out_reason` (helpful for debugging) and return
// false. The cache entry memoizes the lookup result for the lifetime of the
// dataset entry, avoiding the index_statistics open on every plan.
bool VectorIndexMetricMatches(LanceDatasetCacheEntry &entry,
                              const string &column,
                              const string &requested_metric,
                              string &out_reason) {
  auto lookup = entry.GetOrLookupVectorIndexMetric(column);
  if (lookup.status == 1) {
    out_reason = "no vector index on column " + column;
    return false;
  }
  if (lookup.status != 0) {
    out_reason = "vector index metric lookup failed: " + lookup.error_message;
    return false;
  }
  if (lookup.metric != requested_metric) {
    out_reason = "vector index metric '" + lookup.metric +
                 "' does not match requested '" + requested_metric + "'";
    return false;
  }
  return true;
}

unique_ptr<LogicalOperator> RewriteKNNNode(ClientContext &context,
                                           unique_ptr<LogicalOperator> op) {
  if (op->type != LogicalOperatorType::LOGICAL_TOP_N ||
      op->children.size() != 1 || !op->children[0]) {
    return op;
  }
  // The use_index session var is a kill switch for the rewrite as a whole:
  // when off, we leave the plan alone and let DuckDB's TopN drive a normal
  // Lance scan.
  if (!GetKNNOptionBool(context, LANCE_KNN_USE_INDEX_SETTING, true)) {
    return op;
  }
  auto &top_n = op->Cast<LogicalTopN>();

  // --- TopN shape checks ----------------------------------------------------
  if (top_n.orders.size() != 1) {
    return op;
  }
  auto &order = top_n.orders[0];
  if (order.type != OrderType::ASCENDING) {
    return op;
  }
  if (top_n.offset != 0) {
    return op;
  }
  // DuckDB-pin: top_n.limit / top_n.offset are plain idx_t in the vendored
  // submodule. Newer DuckDB wraps both in BoundLimitNode; switch the access
  // pattern if the submodule bump fails to compile here.
  auto configured_max_k = GetKNNOptionUInt64(context, LANCE_KNN_MAX_K_SETTING,
                                             LANCE_KNN_MAX_K_DEFAULT);
  auto effective_max_k =
      configured_max_k == 0 ? LANCE_KNN_MAX_K_DEFAULT : configured_max_k;
  if (effective_max_k > LANCE_KNN_ABSOLUTE_MAX_K) {
    effective_max_k = LANCE_KNN_ABSOLUTE_MAX_K;
  }
  if (top_n.limit == 0 || top_n.limit > effective_max_k) {
    return op;
  }
  if (!order.expression) {
    return op;
  }

  // --- Plan-shape walk: TopN → Projection → [Filter →] GET ------------------
  if (top_n.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
    return op;
  }
  auto &projection = top_n.children[0]->Cast<LogicalProjection>();
  if (projection.children.size() != 1 || !projection.children[0]) {
    return op;
  }

  LogicalFilter *filter_node = nullptr;
  LogicalOperator *get_owner = projection.children[0].get();
  if (get_owner->type == LogicalOperatorType::LOGICAL_FILTER) {
    filter_node = &get_owner->Cast<LogicalFilter>();
    if (filter_node->children.size() != 1 || !filter_node->children[0]) {
      return op;
    }
    get_owner = filter_node->children[0].get();
  }
  if (!get_owner || get_owner->type != LogicalOperatorType::LOGICAL_GET) {
    return op;
  }
  auto &get = get_owner->Cast<LogicalGet>();
  if (!IsLanceScanTableFunction(get.function) || !get.bind_data) {
    return op;
  }
  auto &scan_bind = get.bind_data->Cast<LanceScanBindData>();
  if (!scan_bind.dataset) {
    return op;
  }
  auto prefilter = GetKNNOptionBool(context, LANCE_KNN_PREFILTER_SETTING, true);

  // A scan that already has pushed down LIMIT/OFFSET or take-rowids must not
  // be re-wrapped: that earlier pushdown drives an entirely different scan
  // path. Treating this as a fallback is strictly safe.
  if (scan_bind.limit_offset_pushed_down || !scan_bind.take_row_ids.empty()) {
    return op;
  }

  // If filters are present, prefilter=false would make Lance take the global
  // KNN result first and apply the filter afterwards. SQL's WHERE semantics
  // require filtering before ORDER BY/LIMIT, so leave the original TopN plan
  // in place for this explicitly non-prefiltered mode.
  auto has_filter_source = filter_node ||
                           !scan_bind.lance_pushed_filter_ir_parts.empty() ||
                           !get.table_filters.filters.empty();
  if (!prefilter && has_filter_source) {
    return op;
  }

  // --- Distance call extraction --------------------------------------------
  auto *resolved =
      ResolveThroughProjection(order.expression.get(), &projection);
  if (!resolved) {
    return op;
  }
  DistanceCallInfo info;
  if (!TryDescribeDistanceCall(*resolved, get, info)) {
    return op;
  }

  // --- Vector index existence + metric agreement ---------------------------
  string reason;
  if (!scan_bind.dataset_entry ||
      !VectorIndexMetricMatches(*scan_bind.dataset_entry, info.vector_column,
                                info.metric, reason)) {
    return op;
  }

  // --- Filter IR assembly --------------------------------------------------
  //
  // Three sources feed into the merged IR:
  //   (a) filter_node->expressions               — predicates still live on
  //                                                 the LogicalFilter
  //   (b) scan_bind.lance_pushed_filter_ir_parts — IR fragments injected by
  //                                                 sibling passes (e.g.
  //                                                 LanceLikePushdown)
  //   (c) ProbeLanceTableFilterIR(get, ...)      — table_filters folded out
  //                                                 of the LogicalFilter by
  //                                                 ColumnLifetimeAnalyzer
  //
  // (a) and (b) are *not* disjoint today: LIKE pushdown writes into (b) but
  // leaves the expression in (a) as a DuckDB-side defense in depth. Encoding
  // both would push the same predicate to Lance twice. We dedupe by the
  // serialized IR bytes — the encoder is deterministic for a given
  // (expression, schema), so byte equality implies semantic equality.
  vector<string> filter_parts;
  unordered_set<string> seen_parts;
  auto add_part = [&](string part) {
    if (seen_parts.insert(part).second) {
      filter_parts.push_back(std::move(part));
    }
  };

  if (filter_node) {
    for (auto &expr : filter_node->expressions) {
      if (!expr || expr->HasParameter() || expr->IsVolatile() ||
          expr->CanThrow()) {
        return op;
      }
      string part;
      if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types,
                                     false, *expr, part)) {
        return op;
      }
      add_part(std::move(part));
    }
  }

  for (auto &part : scan_bind.lance_pushed_filter_ir_parts) {
    add_part(part);
  }

  // table_filters source: probe returns parts only if every filter is
  // encodable — anything else means there is a residual DuckDB filter the
  // KNN scan would silently drop, so we fall back instead.
  auto probe = ProbeLanceTableFilterIR(get, scan_bind.names, scan_bind.types);
  if (!probe.all_filters_pushed) {
    return op;
  }
  // Bail if any table_filter targets the rowid column — rowid filters are
  // not part of the KNN IR contract and we'd lose semantics if we erased
  // them silently.
  auto &col_ids = get.GetColumnIds();
  for (auto &entry : get.table_filters.filters) {
    if (entry.first >= col_ids.size()) {
      return op;
    }
    auto col_id = col_ids[entry.first].GetPrimaryIndex();
    if (col_id == COLUMN_IDENTIFIER_ROW_ID) {
      return op;
    }
  }
  for (auto &part : probe.parts) {
    add_part(std::move(part));
  }

  string filter_ir_msg;
  if (!filter_parts.empty()) {
    if (!TryEncodeLanceFilterIRMessage(filter_parts, filter_ir_msg)) {
      return op;
    }
  }

  // --- Build the replacement LogicalGet around __lance_knn_scan ------------
  auto knn_bind = make_uniq<LanceKnnScanBindData>();
  knn_bind->file_path = scan_bind.file_path;
  knn_bind->vector_column = info.vector_column;
  knn_bind->query = std::move(info.query);
  knn_bind->k = NumericCast<uint64_t>(top_n.limit);
  knn_bind->metric = info.metric;
  knn_bind->nprobes = GetKNNOptionUInt64(context, LANCE_KNN_NPROBES_SETTING, 0);
  knn_bind->refine_factor =
      GetKNNOptionUInt64(context, LANCE_KNN_REFINE_FACTOR_SETTING, 0);
  knn_bind->ef = GetKNNOptionUInt64(context, LANCE_KNN_EF_SEARCH_SETTING, 0);
  knn_bind->prefilter = prefilter;
  knn_bind->use_index =
      GetKNNOptionBool(context, LANCE_KNN_USE_INDEX_SETTING, true);
  knn_bind->filter_ir_msg = filter_ir_msg;
  knn_bind->dataset_entry = scan_bind.dataset_entry;
  knn_bind->dataset_cache_hit = scan_bind.dataset_cache_hit;
  // Populate arrow_table / schema_root / names / types from the live Lance
  // KNN schema. The execution path reads bind_data.arrow_table.GetColumns()
  // during ArrowToDuckDB conversion, so we cannot skip this step.
  LanceKnnScanPopulateSchema(context, *knn_bind);

  // The KNN scan schema prepends synthetic columns (_distance, _rowid) ahead
  // of the user's columns, so the upstream column_ids (which index into the
  // original Lance scan schema) must be remapped by name.
  vector<ColumnIndex> remapped_column_ids;
  remapped_column_ids.reserve(get.GetColumnIds().size());
  for (auto &col_id : get.GetColumnIds()) {
    auto p_orig = col_id.GetPrimaryIndex();
    if (p_orig == COLUMN_IDENTIFIER_ROW_ID) {
      // Rowid semantics differ on the KNN path; safer to fall back.
      return op;
    }
    if (p_orig >= scan_bind.names.size()) {
      return op;
    }
    auto &original_name = scan_bind.names[p_orig];
    idx_t p_new = DConstants::INVALID_INDEX;
    for (idx_t i = 0; i < knn_bind->names.size(); i++) {
      if (knn_bind->names[i] == original_name) {
        p_new = i;
        break;
      }
    }
    if (p_new == DConstants::INVALID_INDEX) {
      return op;
    }
    remapped_column_ids.emplace_back(p_new);
  }

  auto returned_types = knn_bind->types;
  auto returned_names = knn_bind->names;

  auto knn_get = make_uniq<LogicalGet>(
      get.table_index, LanceKnnScanFunction(), std::move(knn_bind),
      std::move(returned_types), std::move(returned_names));
  knn_get->SetColumnIds(std::move(remapped_column_ids));
  knn_get->projection_ids = get.projection_ids;
  knn_get->SetEstimatedCardinality(top_n.limit);

  // Parameters for re-bind (prepared statement / EXPLAIN ANALYZE replan).
  // Layout must match LanceKnnScanFunction()'s 11-parameter signature.
  auto &bind_data_ref = knn_get->bind_data->Cast<LanceKnnScanBindData>();
  knn_get->parameters.reserve(11);
  knn_get->parameters.emplace_back(Value(bind_data_ref.file_path));
  knn_get->parameters.emplace_back(Value(bind_data_ref.vector_column));
  string query_blob;
  query_blob.resize(bind_data_ref.query.size() * sizeof(float));
  if (!bind_data_ref.query.empty()) {
    memcpy(&query_blob[0], bind_data_ref.query.data(), query_blob.size());
  }
  knn_get->parameters.emplace_back(Value::BLOB_RAW(query_blob));
  knn_get->parameters.emplace_back(
      Value::BIGINT(NumericCast<int64_t>(bind_data_ref.k)));
  knn_get->parameters.emplace_back(Value(bind_data_ref.metric));
  knn_get->parameters.emplace_back(
      Value::BLOB_RAW(bind_data_ref.filter_ir_msg));
  knn_get->parameters.emplace_back(
      Value::BIGINT(NumericCast<int64_t>(bind_data_ref.nprobes)));
  knn_get->parameters.emplace_back(
      Value::BIGINT(NumericCast<int64_t>(bind_data_ref.refine_factor)));
  knn_get->parameters.emplace_back(Value::BOOLEAN(bind_data_ref.prefilter));
  knn_get->parameters.emplace_back(Value::BOOLEAN(bind_data_ref.use_index));
  knn_get->parameters.emplace_back(
      Value::BIGINT(NumericCast<int64_t>(bind_data_ref.ef)));

  // --- Splice the new GET in place: drop TopN and Filter -------------------
  projection.children[0] = std::move(knn_get);
  projection.SetEstimatedCardinality(top_n.limit);
  // TODO: the surviving projection still computes array_distance(vec, q) on
  // every output row. DuckDB's projection pruner does not eliminate it after
  // our rewrite. Cost is only k rows of one distance call so we tolerate it,
  // but a follow-up could rewrite the projection to read _distance straight
  // from the KNN scan output (the first column it returns).
  auto new_root = std::move(top_n.children[0]);
  return new_root;
}

unique_ptr<LogicalOperator>
LanceKNNScanRewrite(ClientContext &context, unique_ptr<LogicalOperator> op) {
  for (auto &child : op->children) {
    child = LanceKNNScanRewrite(context, std::move(child));
  }
  return RewriteKNNNode(context, std::move(op));
}

void LanceKNNScanRewriteOptimizer(OptimizerExtensionInput &input,
                                  unique_ptr<LogicalOperator> &plan) {
  plan = LanceKNNScanRewrite(input.context, std::move(plan));
}

} // namespace

void RegisterLanceKNNOptimizer(DBConfig &config) {
  OptimizerExtension ext;
  ext.optimize_function = LanceKNNScanRewriteOptimizer;
  OptimizerExtension::Register(config, std::move(ext));
}

} // namespace duckdb
