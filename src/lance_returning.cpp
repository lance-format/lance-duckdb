#include "lance_returning.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_merge_into.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_create.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "lance_scan_bind_data.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "lance_arrow_compat.hpp"
#include "lance_blob.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"
#include "lance_storage.hpp"
#include "lance_write.hpp"

namespace duckdb {

LanceReturningSourceState::LanceReturningSourceState(
    optional_ptr<ColumnDataCollection> rows_p)
    : rows(rows_p) {
  if (rows) {
    rows->InitializeScan(scan);
  }
}

SourceResultType LanceReturningSourceState::Scan(DataChunk &chunk) {
  rows->Scan(scan, chunk);
  return chunk.size() ? SourceResultType::HAVE_MORE_OUTPUT
                      : SourceResultType::FINISHED;
}

static void ConvertReturningArrow(ClientContext &context,
                                  ArrowArrayWrapper &array, ArrowSchema &schema,
                                  DataChunk &out,
                                  const vector<LogicalType> &types) {
  ArrowTableSchema arrow_table;
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table, schema);
  auto owned = make_uniq<ArrowArrayWrapper>();
  owned->arrow_array = array.arrow_array;
  array.arrow_array.release = nullptr;
  ArrowScanLocalState scan_state(std::move(owned), context);
  out.Initialize(context, types);
  out.SetChildCardinality(
      NumericCast<idx_t>(scan_state.chunk->arrow_array.length));
  ArrowTableFunction::ArrowToDuckDB(scan_state, arrow_table.GetColumns(), out,
                                    true, DConstants::INVALID_INDEX);
}

static void ConvertReturningValues(ClientContext &context, DataChunk &values,
                                   const vector<string> &names,
                                   const vector<LogicalType> &types,
                                   DataChunk &out) {
  bool needs_conversion = false;
  for (auto &type : values.GetTypes()) {
    needs_conversion |= LanceIsBlobInput(type);
  }
  if (!needs_conversion) {
    out.Initialize(context, types);
    out.Reference(values);
    return;
  }

  LanceBatchSourceOptions source_options;
  LanceCollectMaterializedBlobSourceOptions(context, values, source_options);
  LanceBlobSourceOptionArrays options;
  BuildLanceBlobSourceOptionArrays(source_options, options);
  auto props = context.GetClientProperties();
  ArrowSchemaWrapper input_schema;
  input_schema.arrow_schema = {};
  ArrowConverter::ToArrowSchema(&input_schema.arrow_schema, values.GetTypes(),
                                names, props);
  ArrowArrayWrapper input_array;
  input_array.arrow_array = {};
  unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>> extensions;
  ArrowConverter::ToArrowArray(values, &input_array.arrow_array, props,
                               extensions);
  ArrowSchemaWrapper output_schema;
  output_schema.arrow_schema = {};
  ArrowArrayWrapper output_array;
  output_array.arrow_array = {};
  auto rc = lance_blob_inputs_to_binary(
      &input_schema.arrow_schema, &input_array.arrow_array, options.Uris(),
      options.KeyOffsets(), options.ValueOffsets(), options.Counts(),
      options.counts.size(),
      NumericCast<uint64_t>(LanceBlobIngestBatchBudget(context)),
      LanceGetSessionHandle(context), &output_schema.arrow_schema,
      &output_array.arrow_array);
  if (rc != 0) {
    throw IOException("Failed to materialize Blob input values for RETURNING" +
                      LanceFormatErrorSuffix());
  }
  ConvertReturningArrow(context, output_array, output_schema.arrow_schema, out,
                        types);
}

void LanceAppendReturningRows(ClientContext &context,
                              ColumnDataCollection &collection,
                              const TableCatalogEntry &table,
                              const vector<bool> &needed, DataChunk &values,
                              const vector<idx_t> &assigned_columns,
                              optional_ptr<LanceDatasetCacheEntry> dataset,
                              const vector<uint64_t> &row_ids,
                              const string &action) {
  const auto &columns = table.GetColumns();
  auto table_types = table.GetTypes();
  vector<idx_t> assigned_to_table;
  vector<idx_t> assigned_to_input;
  vector<string> assigned_names;
  vector<LogicalType> assigned_types;
  vector<LogicalType> input_types;
  vector<bool> assigned(table_types.size(), false);
  for (idx_t i = 0; i < assigned_columns.size(); i++) {
    auto column = assigned_columns[i];
    assigned[column] = true;
    if (!needed[column]) {
      continue;
    }
    assigned_to_table.push_back(column);
    assigned_to_input.push_back(i);
    assigned_names.push_back(
        columns.GetColumn(PhysicalIndex(column)).Name().GetIdentifierName());
    assigned_types.push_back(table_types[column]);
    input_types.push_back(values.data[i].GetType());
  }
  DataChunk selected;
  selected.Initialize(context, input_types);
  selected.SetChildCardinality(values.size());
  for (idx_t i = 0; i < assigned_to_input.size(); i++) {
    selected.data[i].Reference(values.data[assigned_to_input[i]]);
  }
  DataChunk converted;
  ConvertReturningValues(context, selected, assigned_names, assigned_types,
                         converted);

  vector<string> taken_names;
  vector<LogicalType> taken_types;
  vector<idx_t> taken_to_table;
  for (idx_t i = 0; i < table_types.size(); i++) {
    if (needed[i] && !assigned[i]) {
      taken_to_table.push_back(i);
      taken_names.push_back(
          columns.GetColumn(PhysicalIndex(i)).Name().GetIdentifierName());
      taken_types.push_back(table_types[i]);
    }
  }
  auto output_types = table_types;
  if (!action.empty()) {
    output_types.push_back(LogicalType::VARCHAR);
  }
  auto append = [&](optional_ptr<DataChunk> taken, idx_t offset, idx_t count) {
    DataChunk output;
    output.Initialize(context, output_types);
    output.SetChildCardinality(count);
    for (idx_t i = 0; i < table_types.size(); i++) {
      output.data[i].Reference(Value(table_types[i]), count_t(count));
    }
    for (idx_t i = 0; i < taken_to_table.size(); i++) {
      output.data[taken_to_table[i]].Reference(taken->data[i]);
    }
    for (idx_t i = 0; i < assigned_to_table.size(); i++) {
      output.data[assigned_to_table[i]].Slice(converted.data[i], offset,
                                              offset + count);
    }
    if (!action.empty()) {
      output.data[table_types.size()].Reference(Value(action), count_t(count));
    }
    collection.Append(output);
  };
  if (taken_names.empty()) {
    append(nullptr, 0, dataset ? row_ids.size() : values.size());
    return;
  }
  if (!dataset) {
    throw InternalException(
        "Lance RETURNING has unassigned columns without a dataset");
  }
  vector<const char *> name_ptrs;
  for (const auto &name : taken_names) {
    name_ptrs.push_back(name.c_str());
  }
  auto *stream = lance_create_dataset_take_stream_unfiltered(
      dataset->Handle(), row_ids.data(), row_ids.size(), name_ptrs.data(),
      name_ptrs.size());
  if (!stream) {
    throw IOException("Failed to create Lance take stream for RETURNING" +
                      LanceFormatErrorSuffix());
  }
  std::unique_ptr<void, decltype(&lance_close_stream)> owner(
      stream, lance_close_stream);
  idx_t offset = 0;
  while (true) {
    void *batch = nullptr;
    auto rc = lance_stream_next(stream, &batch);
    if (rc == 1) {
      break;
    }
    if (rc != 0) {
      throw IOException("Failed to read Lance take stream for RETURNING" +
                        LanceFormatErrorSuffix());
    }
    LanceArrowBatch exported(batch, "Failed to export Lance RETURNING batch");
    auto &schema = exported.schema.arrow_schema;
    LanceCoerceArrowSchemaForDuckDB(&schema);
    DataChunk taken;
    ConvertReturningArrow(context, *exported.array, schema, taken, taken_types);
    if (taken.size() > row_ids.size() - offset) {
      throw IOException("Lance RETURNING take returned too many rows");
    }
    append(taken, offset, taken.size());
    offset += taken.size();
  }
  if (offset != row_ids.size()) {
    throw IOException("Lance RETURNING take returned an unexpected row count");
  }
}

namespace {

static LogicalGet *FindGetByTableIndex(LogicalOperator &op,
                                       TableIndex table_index) {
  if (op.type == LogicalOperatorType::LOGICAL_GET) {
    auto &get = op.Cast<LogicalGet>();
    if (get.table_index == table_index) {
      return &get;
    }
  }
  for (auto &child : op.children) {
    if (child) {
      auto result = FindGetByTableIndex(*child, table_index);
      if (result) {
        return result;
      }
    }
  }
  return nullptr;
}

static void AddExpressionBindings(const Expression &expression,
                                  column_binding_set_t &required,
                                  vector<ColumnBinding> &worklist) {
  ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
      expression, [&](const BoundColumnRefExpression &ref) {
        if (required.insert(ref.Binding()).second) {
          worklist.push_back(ref.Binding());
        }
      });
}

static void CollectProjectionDefinitions(
    LogicalOperator &op,
    column_binding_map_t<const Expression *> &definitions) {
  if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
    auto &projection = op.Cast<LogicalProjection>();
    for (idx_t i = 0; i < projection.expressions.size(); i++) {
      auto &expression = projection.expressions[i];
      if (expression) {
        definitions.emplace(
            ColumnBinding(projection.table_index, ProjectionIndex(i)),
            expression.get());
      }
    }
  }
  for (auto &child : op.children) {
    if (child) {
      CollectProjectionDefinitions(*child, definitions);
    }
  }
}

static void CollectDmlSeedBindings(LogicalOperator &op,
                                   column_binding_set_t &required,
                                   vector<ColumnBinding> &worklist) {
  // Follow projections only from consumed expressions. MERGE's generic child
  // projection also contains fields that its actions and RETURNING never read.
  if (op.type != LogicalOperatorType::LOGICAL_PROJECTION) {
    LogicalOperatorVisitor::EnumerateExpressions(
        op, [&](unique_ptr<Expression> *expression) {
          if (*expression) {
            AddExpressionBindings(**expression, required, worklist);
          }
        });
  }
  for (auto &child : op.children) {
    if (child) {
      CollectDmlSeedBindings(*child, required, worklist);
    }
  }
}

static column_binding_set_t RequiredDmlBindings(LogicalOperator &op) {
  column_binding_map_t<const Expression *> projection_definitions;
  CollectProjectionDefinitions(op, projection_definitions);

  column_binding_set_t required;
  vector<ColumnBinding> worklist;
  CollectDmlSeedBindings(op, required, worklist);

  for (idx_t i = 0; i < worklist.size(); i++) {
    auto definition = projection_definitions.find(worklist[i]);
    if (definition != projection_definitions.end()) {
      AddExpressionBindings(*definition->second, required, worklist);
    }
  }
  return required;
}

static optional_ptr<LogicalGet>
FindLanceMergeTargetGet(LogicalMergeInto &merge) {
  if (merge.children.empty() || !merge.children[0] ||
      merge.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
    return nullptr;
  }
  auto &child = merge.children[0]->Cast<LogicalProjection>();
  if (merge.row_id_start >= child.expressions.size()) {
    return nullptr;
  }

  column_binding_map_t<const Expression *> projection_definitions;
  CollectProjectionDefinitions(*merge.children[0], projection_definitions);

  ColumnBinding binding(child.table_index, ProjectionIndex(merge.row_id_start));
  column_binding_set_t visited;
  while (visited.insert(binding).second) {
    auto definition = projection_definitions.find(binding);
    if (definition != projection_definitions.end()) {
      vector<ColumnBinding> references;
      ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
          *definition->second, [&](const BoundColumnRefExpression &ref) {
            references.push_back(ref.Binding());
          });
      if (references.size() != 1) {
        return nullptr;
      }
      binding = references[0];
      continue;
    }
    auto *get = FindGetByTableIndex(*merge.children[0], binding.table_index);
    if (!get || get->GetTable().get() != &merge.table) {
      return nullptr;
    }
    return get;
  }
  return nullptr;
}

// DuckDB marks every child column of a DML RETURNING operator as required, so
// a target scan would otherwise read unrelated heavy BLOB columns even when
// neither the mutation nor RETURNING reads them. Preserve all consumed scan
// bindings and replace only unused outputs with typed NULLs.
static void MaskUnusedLanceTargetColumns(LogicalOperator &op, LogicalGet &get,
                                         TableCatalogEntry &table) {
  auto &bind_data = get.bind_data->Cast<LanceScanBindData>();
  auto old_bindings = get.GetColumnBindings();
  auto required = RequiredDmlBindings(op);

  for (idx_t i = 0; i < old_bindings.size(); i++) {
    auto &column_index = get.GetColumnIndex(ProjectionIndex(i));
    auto is_physical_column = column_index.HasPrimaryIndex() &&
                              column_index.GetPrimaryIndex() <
                                  table.GetColumns().PhysicalColumnCount();
    if (is_physical_column && required.count(old_bindings[i]) == 0) {
      bind_data.forced_null_columns.insert(
          NumericCast<column_t>(column_index.GetPrimaryIndex()));
    }
  }
}

// A pass-through logical carrier placed between the SQL RETURNING projection
// and the DML operator. It keeps the full-width row image schema while
// telling the sink which columns the projection references.
class LanceReturningCarrier final : public LogicalExtensionOperator {
public:
  LanceReturningCarrier(unique_ptr<LogicalOperator> child,
                        vector<bool> returning_needed_p)
      : returning_needed(std::move(returning_needed_p)) {
    children.push_back(std::move(child));
  }

  vector<ColumnBinding> GetColumnBindings() override {
    return children[0]->GetColumnBindings();
  }

  void ResolveTypes() override { types = children[0]->types; }

  vector<TableIndex> GetTableIndex() const override {
    return children[0]->GetTableIndex();
  }

  unique_ptr<LogicalOperator> Copy(ClientContext &context) const override {
    return make_uniq<LanceReturningCarrier>(children[0]->Copy(context),
                                            returning_needed);
  }

  PhysicalOperator &CreatePlan(ClientContext &context,
                               PhysicalPlanGenerator &planner) override {
    auto &plan = planner.CreatePlan(*children[0]);
    auto *physical = dynamic_cast<LanceReturningOperator *>(&plan);
    if (!physical) {
      throw InternalException(
          "Lance RETURNING wrapper did not plan a Lance DML operator");
    }
    physical->SetReturningNeeded(returning_needed);
    (void)context;
    return plan;
  }

  optional_ptr<const string>
  GetTypeBindingVerificationIdentifier() const noexcept override {
    return identifier;
  }

  bool SupportSerialization() const override { return false; }

  string GetExtensionName() const override { return "lance_returning"; }

private:
  const string identifier = "lance_returning";
  vector<bool> returning_needed;
};

} // namespace

// External DELETE initially maps table columns to GET positions. Bind the
// requested values as expressions so native join/projection rewrites resolve
// their final sink positions just as they resolve the target row identity.
static void BindLanceDeleteReturning(LogicalDelete &del,
                                     const vector<bool> &needed) {
  auto &row_id = del.expressions[0]->Cast<BoundColumnRefExpression>();
  auto *get =
      FindGetByTableIndex(*del.children[0], row_id.Binding().table_index);
  if (!get || get->GetTable().get() != &del.table) {
    throw InternalException(
        "Lance DELETE RETURNING could not find the target scan");
  }
  for (idx_t i = 0; i < del.return_columns.size(); i++) {
    auto scan_index = del.return_columns[i];
    del.return_columns[i] = DConstants::INVALID_INDEX;
    if (needed[i]) {
      del.return_columns[i] = del.expressions.size();
      del.expressions.push_back(make_uniq<BoundColumnRefExpression>(
          del.table.GetColumns().GetColumn(PhysicalIndex(i)).Type(),
          ColumnBinding(get->table_index, ProjectionIndex(scan_index))));
    }
  }
  MaskUnusedLanceTargetColumns(del, *get, del.table);
}

static void WrapReturningOperator(LogicalOperator &plan) {
  if (plan.type == LogicalOperatorType::LOGICAL_PROJECTION &&
      !plan.children.empty() && plan.children[0]) {
    auto &child = *plan.children[0];
    TableCatalogEntry *table = nullptr;
    TableIndex table_index;
    bool returns = false;
    switch (child.type) {
    case LogicalOperatorType::LOGICAL_INSERT: {
      auto &op = child.Cast<LogicalInsert>();
      table = &op.table;
      table_index = op.table_index;
      returns = op.return_chunk;
      break;
    }
    case LogicalOperatorType::LOGICAL_UPDATE: {
      auto &op = child.Cast<LogicalUpdate>();
      table = &op.table;
      table_index = op.table_index;
      returns = op.return_chunk;
      break;
    }
    case LogicalOperatorType::LOGICAL_DELETE: {
      auto &op = child.Cast<LogicalDelete>();
      table = &op.table;
      table_index = op.table_index;
      returns = op.return_chunk;
      break;
    }
    case LogicalOperatorType::LOGICAL_MERGE_INTO: {
      auto &op = child.Cast<LogicalMergeInto>();
      table = &op.table;
      table_index = op.table_index;
      returns = op.return_chunk;
      break;
    }
    default:
      break;
    }
    if (returns && dynamic_cast<LanceTableEntry *>(table)) {
      vector<bool> needed(child.GetColumnBindings().size(), false);
      for (auto &expression : plan.expressions) {
        ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
            *expression, [&](const BoundColumnRefExpression &ref) {
              if (ref.Binding().table_index == table_index) {
                needed.at(ref.Binding().column_index) = true;
              }
            });
      }
      if (child.type == LogicalOperatorType::LOGICAL_DELETE) {
        BindLanceDeleteReturning(child.Cast<LogicalDelete>(), needed);
      } else if (child.type == LogicalOperatorType::LOGICAL_MERGE_INTO) {
        auto &merge = child.Cast<LogicalMergeInto>();
        auto get = FindLanceMergeTargetGet(merge);
        if (get) {
          MaskUnusedLanceTargetColumns(merge, *get, *table);
        }
      }
      plan.children[0] = make_uniq<LanceReturningCarrier>(
          std::move(plan.children[0]), std::move(needed));
    }
  }
  for (auto &child : plan.children) {
    if (child) {
      WrapReturningOperator(*child);
    }
  }
}

static void CountWriteOperators(LogicalOperator &plan, idx_t &count,
                                bool &writes_lance, bool &has_modifying_cte) {
  TableCatalogEntry *table = nullptr;
  switch (plan.type) {
  case LogicalOperatorType::LOGICAL_INSERT:
    table = &plan.Cast<LogicalInsert>().table;
    break;
  case LogicalOperatorType::LOGICAL_UPDATE:
    table = &plan.Cast<LogicalUpdate>().table;
    break;
  case LogicalOperatorType::LOGICAL_DELETE:
    table = &plan.Cast<LogicalDelete>().table;
    break;
  case LogicalOperatorType::LOGICAL_MERGE_INTO:
    table = &plan.Cast<LogicalMergeInto>().table;
    break;
  case LogicalOperatorType::LOGICAL_CREATE_TABLE:
    if (!plan.children.empty()) {
      count++;
      writes_lance |=
          IsLanceCatalog(plan.Cast<LogicalCreateTable>().schema.catalog);
    }
    break;
  case LogicalOperatorType::LOGICAL_CREATE_TYPE:
    if (!plan.children.empty()) {
      count++;
      auto schema = plan.Cast<LogicalCreate>().schema;
      writes_lance |= schema && IsLanceCatalog(schema->catalog);
    }
    break;
  case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
    count++;
    break;
  case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
    if (IsLanceCopy(plan)) {
      count++;
      writes_lance = true;
    }
    break;
  default:
    break;
  }
  if (table) {
    count++;
    writes_lance |= dynamic_cast<LanceTableEntry *>(table) != nullptr;
  }
  for (idx_t i = 0; i < plan.children.size(); i++) {
    idx_t child_count = 0;
    CountWriteOperators(*plan.children[i], child_count, writes_lance,
                        has_modifying_cte);
    // DuckDB keeps every modifying CTE as a materialized producer at bind
    // time, including unreferenced and NOT MATERIALIZED CTEs.
    if (plan.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && i == 0 &&
        child_count > 0) {
      has_modifying_cte = true;
    }
    count += child_count;
  }
}

void LanceWrapReturningOperator(LogicalOperator &plan) {
  idx_t count = 0;
  bool writes_lance = false;
  bool has_modifying_cte = false;
  CountWriteOperators(plan, count, writes_lance, has_modifying_cte);
  // DuckDB's CTE capability check does not count an outer write consumer.
  // Reject the complete statement before any operator can stage a write.
  // Generated multi-table plans such as COPY DATABASE have no modifying CTE.
  if (has_modifying_cte && writes_lance && count > 1) {
    throw NotImplementedException(
        "Multiple writes in a single query involving Lance are not "
        "supported");
  }
  WrapReturningOperator(plan);
}

} // namespace duckdb
