#include "lance_write.hpp"

#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/parser/statement/copy_statement.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "lance_arrow_compat.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_session_state.hpp"

#include <cstdint>

namespace duckdb {

struct LanceWriteBindData : public FunctionData {
  string mode = "create";
  bool write_empty_file = true;
  string data_storage_version = LANCE_DEFAULT_DATA_STORAGE_VERSION;
  uint64_t max_rows_per_file = LANCE_DEFAULT_MAX_ROWS_PER_FILE;
  uint64_t max_rows_per_group = LANCE_DEFAULT_MAX_ROWS_PER_GROUP;
  uint64_t max_bytes_per_file = LANCE_DEFAULT_MAX_BYTES_PER_FILE;

  vector<string> names;
  vector<LogicalType> types;

  unique_ptr<FunctionData> Copy() const override {
    auto result = make_uniq<LanceWriteBindData>();
    result->mode = mode;
    result->write_empty_file = write_empty_file;
    result->data_storage_version = data_storage_version;
    result->max_rows_per_file = max_rows_per_file;
    result->max_rows_per_group = max_rows_per_group;
    result->max_bytes_per_file = max_bytes_per_file;
    result->names = names;
    result->types = types;
    return std::move(result);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceWriteBindData>();
    return mode == other.mode && write_empty_file == other.write_empty_file &&
           data_storage_version == other.data_storage_version &&
           max_rows_per_file == other.max_rows_per_file &&
           max_rows_per_group == other.max_rows_per_group &&
           max_bytes_per_file == other.max_bytes_per_file &&
           names == other.names && types == other.types;
  }
};

struct LanceWriteGlobalState : public GlobalFunctionData {
  explicit LanceWriteGlobalState() = default;

  string cache_key;
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  void *writer = nullptr;
  ArrowSchemaWrapper schema_root;

  ~LanceWriteGlobalState() override {
    if (writer) {
      lance_close_writer(writer);
      writer = nullptr;
    }
  }
};

struct LanceWriteLocalState : public LocalFunctionData {};

static void LanceWriteCopyOptions(ClientContext &, CopyOptionsInput &input) {
  auto &options = input.options;
  options["mode"] =
      CopyOption(LogicalType::VARCHAR, CopyOptionMode::WRITE_ONLY);
  options["data_storage_version"] =
      CopyOption(LogicalType::VARCHAR, CopyOptionMode::WRITE_ONLY);
  options["max_rows_per_file"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
  options["max_rows_per_group"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
  options["max_bytes_per_file"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
}

static unique_ptr<FunctionData>
LanceWriteBind(ClientContext &context, CopyFunctionBindInput &input,
               const vector<Identifier> &names,
               const vector<LogicalType> &sql_types) {
  auto bind_data = make_uniq<LanceWriteBindData>();
  for (const auto &name : names) {
    bind_data->names.push_back(name.GetIdentifierName());
  }
  bind_data->types = sql_types;

  for (auto &option : input.info.options) {
    const auto key = StringUtil::Lower(option.first.GetIdentifierName());
    if (option.second.size() != 1) {
      throw BinderException("%s requires exactly one argument",
                            StringUtil::Upper(key));
    }
    auto &value = option.second[0];
    if (value.IsNull()) {
      throw BinderException("%s cannot be NULL", key);
    }

    if (key == "mode") {
      auto mode = StringUtil::Lower(value.ToString());
      if (mode != "create" && mode != "append" && mode != "overwrite") {
        throw BinderException(
            "mode must be one of [create, append, overwrite]");
      }
      bind_data->mode = std::move(mode);
    } else if (key == "write_empty_file") {
      bind_data->write_empty_file =
          value.CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
    } else if (key == "data_storage_version") {
      bind_data->data_storage_version =
          value.CastAs(context, LogicalType::VARCHAR).GetValue<string>();
      if (bind_data->data_storage_version.empty()) {
        throw BinderException("data_storage_version cannot be empty");
      }
    } else if (key == "max_rows_per_file") {
      bind_data->max_rows_per_file =
          value.CastAs(context, LogicalType::UBIGINT).GetValue<uint64_t>();
    } else if (key == "max_rows_per_group") {
      bind_data->max_rows_per_group =
          value.CastAs(context, LogicalType::UBIGINT).GetValue<uint64_t>();
    } else if (key == "max_bytes_per_file") {
      bind_data->max_bytes_per_file =
          value.CastAs(context, LogicalType::UBIGINT).GetValue<uint64_t>();
    } else {
      throw BinderException("Unsupported option for COPY FORMAT lance: %s",
                            key);
    }
  }

  return std::move(bind_data);
}

static unique_ptr<GlobalFunctionData>
LanceWriteInitGlobal(ClientContext &context, FunctionData &bind_data_p,
                     const string &file_path) {
  auto &bind_data = bind_data_p.Cast<LanceWriteBindData>();
  auto state = make_uniq<LanceWriteGlobalState>();

  auto props = context.GetClientProperties();
  memset(&state->schema_root.arrow_schema, 0,
         sizeof(state->schema_root.arrow_schema));
  ArrowConverter::ToArrowSchema(&state->schema_root.arrow_schema,
                                bind_data.types, bind_data.names, props);
  LanceNormalizeArrowListFieldNames(&state->schema_root.arrow_schema);

  auto &option_keys = state->option_keys;
  auto &option_values = state->option_values;
  auto &open_path = state->open_path;
  ResolveLanceStorageOptions(context, file_path, open_path, option_keys,
                             option_values);
  state->cache_key = LanceBuildResolvedPathDatasetCacheKey(
      open_path, option_keys, option_values);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);

  const char *data_storage_version_ptr =
      bind_data.data_storage_version.empty()
          ? nullptr
          : bind_data.data_storage_version.c_str();
  auto *session = LanceGetSessionHandle(context);
  state->writer = lance_open_uncommitted_writer_with_storage_options(
      open_path.c_str(), bind_data.mode.c_str(),
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      bind_data.max_rows_per_file, bind_data.max_rows_per_group,
      bind_data.max_bytes_per_file, data_storage_version_ptr, session,
      &state->schema_root.arrow_schema);
  if (!state->writer) {
    throw IOException("Failed to open Lance writer: " + open_path +
                      LanceFormatErrorSuffix());
  }

  return std::move(state);
}

static void LanceWriteSink(ExecutionContext &context, FunctionData &,
                           GlobalFunctionData &gstate_p, LocalFunctionData &,
                           DataChunk &input) {
  if (input.size() == 0) {
    return;
  }

  auto &gstate = gstate_p.Cast<LanceWriteGlobalState>();

  auto props = context.client.GetClientProperties();
  unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
      extension_type_cast;

  ArrowArray array;
  memset(&array, 0, sizeof(array));
  ArrowConverter::ToArrowArray(input, &array, props, extension_type_cast);

  auto rc = lance_writer_write_batch(gstate.writer, &array);
  if (array.release) {
    array.release(&array);
  }
  if (rc != 0) {
    throw IOException("Failed to write to Lance dataset" +
                      LanceFormatErrorSuffix());
  }
}

static void LanceWriteFinalize(ClientContext &context, FunctionData &,
                               GlobalFunctionData &gstate_p) {
  auto &gstate = gstate_p.Cast<LanceWriteGlobalState>();
  if (!gstate.writer) {
    return;
  }
  void *transaction = nullptr;
  auto rc = lance_writer_finish_uncommitted(gstate.writer, &transaction);
  lance_close_writer(gstate.writer);
  gstate.writer = nullptr;
  if (rc != 0) {
    throw IOException("Failed to finalize Lance dataset write" +
                      LanceFormatErrorSuffix());
  }
  vector<const char *> keys;
  vector<const char *> values;
  BuildStorageOptionPointerArrays(gstate.option_keys, gstate.option_values,
                                  keys, values);
  rc = lance_commit_transaction_with_storage_options(
      gstate.open_path.c_str(), keys.empty() ? nullptr : keys.data(),
      values.empty() ? nullptr : values.data(), keys.size(),
      LanceGetSessionHandle(context), transaction);
  if (rc != 0) {
    throw IOException("Failed to commit Lance dataset write" +
                      LanceFormatErrorSuffix());
  }
  LanceInvalidateDatasetCache(context, gstate.cache_key);
}

// Lance commits a dataset manifest, rather than a DuckDB-managed output file.
// Keep the writer alive until finalize; destruction after a failed pipeline
// closes it without publishing the pending write or deleting an existing
// dataset.
struct LanceCopyGlobalState : public GlobalSinkState {
  unique_ptr<GlobalFunctionData> writer;
  idx_t count = 0;
};

class PhysicalLanceCopy final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceCopy(PhysicalPlan &plan, string path_p,
                    unique_ptr<FunctionData> bind_data_p, idx_t cardinality)
      : PhysicalOperator(plan, TYPE, {LogicalType::BIGINT}, cardinality),
        path(std::move(path_p)), bind_data(std::move(bind_data_p)) {}

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return false; }
  bool SinkOrderDependent() const override { return true; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &context) const override {
    auto state = make_uniq<LanceCopyGlobalState>();
    state->writer = LanceWriteInitGlobal(context, *bind_data, path);
    return std::move(state);
  }

  SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                      OperatorSinkInput &input) const override {
    auto &state = input.global_state.Cast<LanceCopyGlobalState>();
    LanceWriteLocalState local;
    LanceWriteSink(context, *bind_data, *state.writer, local, chunk);
    state.count += chunk.size();
    return SinkResultType::NEED_MORE_INPUT;
  }

  SinkCombineResultType Combine(ExecutionContext &,
                                OperatorSinkCombineInput &) const override {
    return SinkCombineResultType::FINISHED;
  }

  SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context,
                            OperatorSinkFinalizeInput &input) const override {
    auto &state = input.global_state.Cast<LanceCopyGlobalState>();
    if (state.count > 0 ||
        bind_data->Cast<LanceWriteBindData>().write_empty_file) {
      LanceWriteFinalize(context, *bind_data, *state.writer);
    }
    return SinkFinalizeType::READY;
  }

  SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                   OperatorSourceInput &) const override {
    chunk.SetCardinality(1);
    chunk.SetValue(0, 0,
                   Value::BIGINT(NumericCast<int64_t>(
                       sink_state->Cast<LanceCopyGlobalState>().count)));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceCopy"; }

private:
  string path;
  unique_ptr<FunctionData> bind_data;
};

PhysicalOperator &PlanLanceCopy(ClientContext &context,
                                PhysicalPlanGenerator &planner, CopyInfo &info,
                                const vector<Identifier> &names,
                                const vector<LogicalType> &types,
                                PhysicalOperator &child) {
  CopyFunctionBindInput input(info);
  auto &copy = planner.Make<PhysicalLanceCopy>(
      info.file_path, LanceWriteBind(context, input, names, types),
      child.estimated_cardinality);
  copy.children.push_back(child);
  return copy;
}

class LogicalLanceCopy final : public LogicalExtensionOperator {
public:
  LogicalLanceCopy(TableIndex index_p, unique_ptr<CopyInfo> info_p,
                   vector<Identifier> names_p, vector<LogicalType> types_p)
      : index(index_p), info(std::move(info_p)), names(std::move(names_p)),
        input_types(std::move(types_p)) {}

  vector<ColumnBinding> GetColumnBindings() override {
    return {ColumnBinding(index, ProjectionIndex(0))};
  }
  vector<TableIndex> GetTableIndex() const override { return {index}; }
  string GetName() const override { return "LanceCopy"; }
  string GetExtensionName() const override { return "lance"; }
  PhysicalOperator &CreatePlan(ClientContext &context,
                               PhysicalPlanGenerator &planner) override {
    auto &child = planner.CreatePlan(*children[0]);
    return PlanLanceCopy(context, planner, *info, names, input_types, child);
  }

protected:
  void ResolveTypes() override { types = {LogicalType::BIGINT}; }

private:
  TableIndex index;
  unique_ptr<CopyInfo> info;
  vector<Identifier> names;
  vector<LogicalType> input_types;
};

static BoundStatement BindLanceCopy(Binder &binder, CopyStatement &statement) {
  auto &info = *statement.info;
  // Reject generic file-layout options: one COPY writes one Lance dataset.
  // Validate before execution, including on empty inputs.
  auto node = info.select_statement->Copy();
  auto select = binder.Bind(*node);
  CopyFunctionBindInput input(info);
  (void)LanceWriteBind(binder.context, input, select.names, select.types);
  auto copy = make_uniq<LogicalLanceCopy>(
      binder.GenerateTableIndex(), info.Copy(), select.names, select.types);
  copy->children.push_back(std::move(select.plan));
  BoundStatement result;
  result.names = {Identifier("Count")};
  result.types = {LogicalType::BIGINT};
  result.plan = std::move(copy);
  binder.GetStatementProperties().return_type =
      StatementReturnType::CHANGED_ROWS;
  return result;
}

void RegisterLanceWrite(ExtensionLoader &loader) {
  CopyFunction function("lance");
  function.extension = "lance";
  function.copy_options = LanceWriteCopyOptions;
  function.plan = BindLanceCopy;
  loader.RegisterFunction(function);
}

} // namespace duckdb
