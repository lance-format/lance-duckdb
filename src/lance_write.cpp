#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"

#include <cstdint>

namespace duckdb {

static constexpr uint64_t DEFAULT_MAX_ROWS_PER_FILE = 1024ULL * 1024ULL;
static constexpr uint64_t DEFAULT_MAX_ROWS_PER_GROUP = 1024ULL;
static constexpr uint64_t DEFAULT_MAX_BYTES_PER_FILE =
    90ULL * 1024ULL * 1024ULL * 1024ULL;

struct LanceWriteBindData : public FunctionData {
  string mode = "create";
  uint64_t max_rows_per_file = DEFAULT_MAX_ROWS_PER_FILE;
  uint64_t max_rows_per_group = DEFAULT_MAX_ROWS_PER_GROUP;
  uint64_t max_bytes_per_file = DEFAULT_MAX_BYTES_PER_FILE;

  vector<string> names;
  vector<LogicalType> types;

  unique_ptr<FunctionData> Copy() const override {
    auto result = make_uniq<LanceWriteBindData>();
    result->mode = mode;
    result->max_rows_per_file = max_rows_per_file;
    result->max_rows_per_group = max_rows_per_group;
    result->max_bytes_per_file = max_bytes_per_file;
    result->names = names;
    result->types = types;
    return std::move(result);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceWriteBindData>();
    return mode == other.mode && max_rows_per_file == other.max_rows_per_file &&
           max_rows_per_group == other.max_rows_per_group &&
           max_bytes_per_file == other.max_bytes_per_file &&
           names == other.names && types == other.types;
  }
};

struct LanceWriteGlobalState : public GlobalFunctionData {
  explicit LanceWriteGlobalState() = default;

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
  options["max_rows_per_file"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
  options["max_rows_per_group"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
  options["max_bytes_per_file"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
}

static unique_ptr<FunctionData>
LanceWriteBind(ClientContext &, CopyFunctionBindInput &input,
               const vector<string> &names,
               const vector<LogicalType> &sql_types) {
  auto bind_data = make_uniq<LanceWriteBindData>();
  bind_data->names = names;
  bind_data->types = sql_types;

  for (auto &option : input.info.options) {
    const auto key = StringUtil::Lower(option.first);
    if (option.second.size() != 1) {
      throw BinderException("%s requires exactly one argument",
                            StringUtil::Upper(key));
    }
    auto &value = option.second[0];

    if (key == "mode") {
      if (value.IsNull()) {
        throw BinderException("mode cannot be NULL");
      }
      auto mode = StringUtil::Lower(value.ToString());
      if (mode != "create" && mode != "append" && mode != "overwrite") {
        throw BinderException(
            "mode must be one of [create, append, overwrite]");
      }
      bind_data->mode = std::move(mode);
    } else if (key == "max_rows_per_file") {
      bind_data->max_rows_per_file = value.GetValue<uint64_t>();
    } else if (key == "max_rows_per_group") {
      bind_data->max_rows_per_group = value.GetValue<uint64_t>();
    } else if (key == "max_bytes_per_file") {
      bind_data->max_bytes_per_file = value.GetValue<uint64_t>();
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

  auto open_path = file_path;
  vector<string> option_keys;
  vector<string> option_values;

  if (StringUtil::StartsWith(open_path, "s3://") ||
      StringUtil::StartsWith(open_path, "s3a://") ||
      StringUtil::StartsWith(open_path, "s3n://")) {
    open_path = LanceNormalizeS3Scheme(open_path);
    LanceFillS3StorageOptionsFromSecrets(context, open_path, option_keys,
                                         option_values);
  }

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  key_ptrs.reserve(option_keys.size());
  value_ptrs.reserve(option_values.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    key_ptrs.push_back(option_keys[i].c_str());
    value_ptrs.push_back(option_values[i].c_str());
  }

  state->writer = lance_open_writer_with_storage_options(
      open_path.c_str(), bind_data.mode.c_str(),
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      bind_data.max_rows_per_file, bind_data.max_rows_per_group,
      bind_data.max_bytes_per_file, &state->schema_root.arrow_schema);
  if (!state->writer) {
    throw IOException("Failed to open Lance writer: " + open_path +
                      LanceFormatErrorSuffix());
  }

  return std::move(state);
}

static unique_ptr<LocalFunctionData> LanceWriteInitLocal(ExecutionContext &,
                                                         FunctionData &) {
  return make_uniq<LanceWriteLocalState>();
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

static void LanceWriteFinalize(ClientContext &, FunctionData &,
                               GlobalFunctionData &gstate_p) {
  auto &gstate = gstate_p.Cast<LanceWriteGlobalState>();
  if (!gstate.writer) {
    return;
  }
  auto rc = lance_writer_finish(gstate.writer);
  lance_close_writer(gstate.writer);
  gstate.writer = nullptr;
  if (rc != 0) {
    throw IOException("Failed to finalize Lance dataset write" +
                      LanceFormatErrorSuffix());
  }
}

static CopyFunctionExecutionMode
LanceWriteExecutionMode(bool preserve_insertion_order, bool) {
  if (!preserve_insertion_order) {
    return CopyFunctionExecutionMode::PARALLEL_COPY_TO_FILE;
  }
  return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
}

void RegisterLanceWrite(ExtensionLoader &loader) {
  CopyFunction function("lance");
  function.extension = "lance";

  function.copy_options = LanceWriteCopyOptions;
  function.copy_to_bind = LanceWriteBind;
  function.copy_to_initialize_global = LanceWriteInitGlobal;
  function.copy_to_initialize_local = LanceWriteInitLocal;
  function.copy_to_sink = LanceWriteSink;
  function.copy_to_finalize = LanceWriteFinalize;
  function.execution_mode = LanceWriteExecutionMode;

  loader.RegisterFunction(function);
}

} // namespace duckdb
