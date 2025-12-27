#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_table_entry.hpp"

#include <cstdint>

namespace duckdb {

struct LanceInsertGlobalState : public GlobalSinkState {
  explicit LanceInsertGlobalState(string dataset_uri_p,
                                  vector<string> column_names_p,
                                  vector<LogicalType> column_types_p)
      : dataset_uri(std::move(dataset_uri_p)),
        column_names(std::move(column_names_p)),
        column_types(std::move(column_types_p)) {}

  mutex lock;

  string dataset_uri;
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;

  vector<string> column_names;
  vector<LogicalType> column_types;

  idx_t insert_count = 0;

  void *writer = nullptr;
  ArrowSchemaWrapper schema_root;

  ~LanceInsertGlobalState() override {
    if (writer) {
      lance_close_writer(writer);
      writer = nullptr;
    }
  }
};

struct LanceInsertLocalState : public LocalSinkState {};

class PhysicalLanceInsert final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceInsert(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                      LanceTableEntry &table_p, string dataset_uri_p,
                      vector<string> column_names_p,
                      vector<LogicalType> column_types_p,
                      idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), dataset_uri(std::move(dataset_uri_p)),
        column_names(std::move(column_names_p)),
        column_types(std::move(column_types_p)) {}

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return false; }
  bool SinkOrderDependent() const override { return false; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &) const override {
    return make_uniq<LanceInsertGlobalState>(dataset_uri, column_names,
                                             column_types);
  }

  unique_ptr<LocalSinkState>
  GetLocalSinkState(ExecutionContext &) const override {
    return make_uniq<LanceInsertLocalState>();
  }

  SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                      OperatorSinkInput &input) const override {
    if (chunk.size() == 0) {
      return SinkResultType::NEED_MORE_INPUT;
    }

    auto &gstate = input.global_state.Cast<LanceInsertGlobalState>();
    lock_guard<mutex> guard(gstate.lock);

    if (!gstate.writer) {
      auto props = context.client.GetClientProperties();
      memset(&gstate.schema_root.arrow_schema, 0,
             sizeof(gstate.schema_root.arrow_schema));
      ArrowConverter::ToArrowSchema(&gstate.schema_root.arrow_schema,
                                    gstate.column_types, gstate.column_names,
                                    props);

      ResolveLanceStorageOptions(context.client, gstate.dataset_uri,
                                 gstate.open_path, gstate.option_keys,
                                 gstate.option_values);

      vector<const char *> key_ptrs;
      vector<const char *> value_ptrs;
      BuildStorageOptionPointerArrays(gstate.option_keys, gstate.option_values,
                                      key_ptrs, value_ptrs);

      gstate.writer = lance_open_uncommitted_writer_with_storage_options(
          gstate.open_path.c_str(), "append",
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(),
          gstate.option_keys.size(), LANCE_DEFAULT_MAX_ROWS_PER_FILE,
          LANCE_DEFAULT_MAX_ROWS_PER_GROUP, LANCE_DEFAULT_MAX_BYTES_PER_FILE,
          &gstate.schema_root.arrow_schema);
      if (!gstate.writer) {
        throw IOException("Failed to open Lance writer: " + gstate.open_path +
                          LanceFormatErrorSuffix());
      }
    }

    unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
        extension_type_cast;
    auto props = context.client.GetClientProperties();

    ArrowArray array;
    memset(&array, 0, sizeof(array));
    ArrowConverter::ToArrowArray(chunk, &array, props, extension_type_cast);

    auto rc = lance_writer_write_batch(gstate.writer, &array);
    if (array.release) {
      array.release(&array);
    }
    if (rc != 0) {
      throw IOException("Failed to write to Lance dataset" +
                        LanceFormatErrorSuffix());
    }
    gstate.insert_count += chunk.size();
    return SinkResultType::NEED_MORE_INPUT;
  }

  SinkCombineResultType Combine(ExecutionContext &,
                                OperatorSinkCombineInput &) const override {
    return SinkCombineResultType::FINISHED;
  }

  SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context,
                            OperatorSinkFinalizeInput &input) const override {
    auto &gstate = input.global_state.Cast<LanceInsertGlobalState>();
    void *txn = nullptr;

    {
      lock_guard<mutex> guard(gstate.lock);
      if (!gstate.writer) {
        return SinkFinalizeType::READY;
      }
      auto rc = lance_writer_finish_uncommitted(gstate.writer, &txn);
      lance_close_writer(gstate.writer);
      gstate.writer = nullptr;
      if (rc != 0 || !txn) {
        throw IOException("Failed to finalize Lance append transaction" +
                          LanceFormatErrorSuffix());
      }
    }

    RegisterLancePendingAppend(
        context, table.ParentCatalog(), std::move(gstate.open_path),
        std::move(gstate.option_keys), std::move(gstate.option_values), txn);
    return SinkFinalizeType::READY;
  }

  class LanceInsertSourceState : public GlobalSourceState {
  public:
    bool emitted = false;
  };

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceInsertSourceState>();
  }

  SourceResultType GetData(ExecutionContext &, DataChunk &chunk,
                           OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceInsertSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    auto &gstate = sink_state->Cast<LanceInsertGlobalState>();
    chunk.SetCardinality(1);
    chunk.SetValue(0, 0,
                   Value::BIGINT(NumericCast<int64_t>(gstate.insert_count)));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceInsert"; }

private:
  LanceTableEntry &table;
  string dataset_uri;
  vector<string> column_names;
  vector<LogicalType> column_types;
};

PhysicalOperator &PlanLanceInsertAppend(ClientContext &context,
                                        PhysicalPlanGenerator &planner,
                                        LogicalInsert &op,
                                        optional_ptr<PhysicalOperator> plan) {
  if (op.return_chunk) {
    throw NotImplementedException(
        "Lance INSERT does not support RETURNING yet");
  }
  if (op.on_conflict_info.action_type != OnConflictAction::THROW) {
    throw NotImplementedException("Lance INSERT does not support ON CONFLICT");
  }
  if (!plan) {
    throw InternalException("Lance INSERT requires a child plan");
  }

  if (!op.column_index_map.empty()) {
    plan = planner.ResolveDefaultsProjection(op, *plan);
  }

  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException("PlanLanceInsertAppend called for non-Lance table");
  }

  vector<string> column_names;
  vector<LogicalType> column_types;
  for (auto &col : op.table.GetColumns().Physical()) {
    column_names.push_back(col.Name());
    column_types.push_back(col.Type());
  }

  auto &insert = planner.Make<PhysicalLanceInsert>(
      op.types, *lance_table, lance_table->DatasetUri(),
      std::move(column_names), std::move(column_types),
      op.estimated_cardinality);
  insert.children.push_back(*plan);
  return insert;
}

} // namespace duckdb
