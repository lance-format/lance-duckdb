#include "duckdb.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/qualified_name.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_table_entry.hpp"

namespace duckdb {

enum class LanceKvTarget : uint8_t {
  CONFIG = 1,
  TABLE_METADATA = 2,
  SCHEMA_METADATA = 3,
  FIELD_METADATA = 4,
  INDICES = 5,
};

static string ResolveLanceDatasetUriFromTableName(ClientContext &context,
                                                  const Value &table_name) {
  if (table_name.IsNull()) {
    throw BinderException("table name cannot be NULL");
  }
  auto table_name_str = table_name.GetValue<string>();
  if (table_name_str.empty()) {
    throw BinderException("table name cannot be empty");
  }

  auto qname = QualifiedName::Parse(table_name_str);
  auto &entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY,
                                  qname.catalog, qname.schema, qname.name);
  auto &table_entry = entry.Cast<TableCatalogEntry>();
  auto *lance_entry = dynamic_cast<LanceTableEntry *>(&table_entry);
  if (!lance_entry) {
    throw NotImplementedException(
        "This operation is only supported for tables in ATTACH TYPE LANCE "
        "directory namespaces");
  }
  return lance_entry->DatasetUri();
}

static vector<pair<string, string>> ParseTsvRows(const char *ptr) {
  if (!ptr) {
    throw IOException("Failed to read metadata from Lance dataset" +
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

struct LanceKvUpdateBindData final : public FunctionData {
  LanceKvUpdateBindData(string dataset_uri_p, LanceKvTarget target_p,
                        string key_p, bool has_value_p, string value_p,
                        string field_path_p, string index_column_p,
                        string index_name_p)
      : dataset_uri(std::move(dataset_uri_p)), target(target_p),
        key(std::move(key_p)), has_value(has_value_p),
        value(std::move(value_p)), field_path(std::move(field_path_p)),
        index_column(std::move(index_column_p)),
        index_name(std::move(index_name_p)) {}

  string dataset_uri;
  LanceKvTarget target;
  string key;
  bool has_value = false;
  string value;
  string field_path;
  string index_column;
  string index_name;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceKvUpdateBindData>(dataset_uri, target, key, has_value,
                                            value, field_path, index_column,
                                            index_name);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceKvUpdateBindData>();
    return dataset_uri == other.dataset_uri && target == other.target &&
           key == other.key && has_value == other.has_value &&
           value == other.value && field_path == other.field_path &&
           index_column == other.index_column && index_name == other.index_name;
  }
};

struct LanceKvListBindData final : public FunctionData {
  LanceKvListBindData(string dataset_uri_p, LanceKvTarget target_p,
                      string field_path_p)
      : dataset_uri(std::move(dataset_uri_p)), target(target_p),
        field_path(std::move(field_path_p)) {}

  string dataset_uri;
  LanceKvTarget target;
  string field_path;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceKvListBindData>(dataset_uri, target, field_path);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceKvListBindData>();
    return dataset_uri == other.dataset_uri && target == other.target &&
           field_path == other.field_path;
  }
};

struct LanceSingleRowGlobalState final : public GlobalTableFunctionState {
  bool finished = false;
};

static unique_ptr<GlobalTableFunctionState>
LanceSingleRowInitGlobal(ClientContext &, TableFunctionInitInput &) {
  return make_uniq<LanceSingleRowGlobalState>();
}

static void LanceKvUpdateFunc(ClientContext &context, TableFunctionInput &data,
                              DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceSingleRowGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceKvUpdateBindData>();
  void *dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + bind_data.dataset_uri +
                      LanceFormatErrorSuffix());
  }

  const char *value_ptr =
      bind_data.has_value ? bind_data.value.c_str() : nullptr;

  int32_t rc = 0;
  switch (bind_data.target) {
  case LanceKvTarget::CONFIG:
    rc = lance_dataset_update_config(dataset, bind_data.key.c_str(), value_ptr);
    break;
  case LanceKvTarget::TABLE_METADATA:
    rc = lance_dataset_update_table_metadata(dataset, bind_data.key.c_str(),
                                             value_ptr);
    break;
  case LanceKvTarget::SCHEMA_METADATA:
    rc = lance_dataset_update_schema_metadata(dataset, bind_data.key.c_str(),
                                              value_ptr);
    break;
  case LanceKvTarget::FIELD_METADATA:
    rc = lance_dataset_update_field_metadata(dataset,
                                             bind_data.field_path.c_str(),
                                             bind_data.key.c_str(), value_ptr);
    break;
  case LanceKvTarget::INDICES:
    rc = lance_dataset_create_scalar_index(dataset,
                                           bind_data.index_column.c_str(),
                                           bind_data.index_name.c_str(), 1);
    break;
  default:
    rc = -1;
    break;
  }

  lance_close_dataset(dataset);
  if (rc != 0) {
    throw IOException("Failed to update Lance dataset: " +
                      bind_data.dataset_uri + LanceFormatErrorSuffix());
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BIGINT(1));
}

static unique_ptr<FunctionData>
LanceKvUpdateBind(ClientContext &context, TableFunctionBindInput &input,
                  vector<LogicalType> &return_types, vector<string> &names,
                  LanceKvTarget target, idx_t expected_inputs) {
  if (input.inputs.size() != expected_inputs) {
    throw BinderException("invalid argument count");
  }

  auto dataset_uri =
      ResolveLanceDatasetUriFromTableName(context, input.inputs[0]);

  string key;
  bool has_value = false;
  string value;
  string field_path;
  string index_column;
  string index_name;

  if (target == LanceKvTarget::INDICES) {
    if (input.inputs[1].IsNull() || input.inputs[2].IsNull()) {
      throw BinderException("index column and index name cannot be NULL");
    }
    index_column = input.inputs[1].GetValue<string>();
    index_name = input.inputs[2].GetValue<string>();
  } else {
    if (input.inputs.size() >= 2 && !input.inputs[1].IsNull()) {
      key = input.inputs[1].GetValue<string>();
    }
    if (key.empty() && (target == LanceKvTarget::CONFIG ||
                        target == LanceKvTarget::TABLE_METADATA ||
                        target == LanceKvTarget::SCHEMA_METADATA)) {
      throw BinderException("key cannot be empty");
    }
  }

  if (target == LanceKvTarget::FIELD_METADATA) {
    if (input.inputs.size() < 3) {
      throw BinderException("invalid argument count");
    }
    if (input.inputs[1].IsNull() ||
        input.inputs[1].GetValue<string>().empty()) {
      throw BinderException("column name cannot be NULL or empty");
    }
    field_path = input.inputs[1].GetValue<string>();
    if (input.inputs[2].IsNull() ||
        input.inputs[2].GetValue<string>().empty()) {
      throw BinderException("key cannot be NULL or empty");
    }
    key = input.inputs[2].GetValue<string>();
    if (input.inputs.size() == 4 && !input.inputs[3].IsNull()) {
      has_value = true;
      value = input.inputs[3].GetValue<string>();
    }
  } else if (target != LanceKvTarget::INDICES) {
    if (input.inputs.size() == 3 && !input.inputs[2].IsNull()) {
      has_value = true;
      value = input.inputs[2].GetValue<string>();
    }
  }

  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  return make_uniq<LanceKvUpdateBindData>(
      std::move(dataset_uri), target, std::move(key), has_value,
      std::move(value), std::move(field_path), std::move(index_column),
      std::move(index_name));
}

struct LanceKvListGlobalState final : public GlobalTableFunctionState {
  explicit LanceKvListGlobalState(vector<pair<string, string>> rows_p)
      : rows(std::move(rows_p)) {}

  vector<pair<string, string>> rows;
  idx_t offset = 0;
};

static unique_ptr<GlobalTableFunctionState>
LanceKvListInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceKvListBindData>();
  void *dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + bind_data.dataset_uri +
                      LanceFormatErrorSuffix());
  }

  const char *ptr = nullptr;
  switch (bind_data.target) {
  case LanceKvTarget::CONFIG:
    ptr = lance_dataset_list_config(dataset);
    break;
  case LanceKvTarget::TABLE_METADATA:
    ptr = lance_dataset_list_table_metadata(dataset);
    break;
  case LanceKvTarget::SCHEMA_METADATA:
    ptr = lance_dataset_list_schema_metadata(dataset);
    break;
  case LanceKvTarget::FIELD_METADATA:
    ptr = lance_dataset_list_field_metadata(dataset,
                                            bind_data.field_path.c_str());
    break;
  case LanceKvTarget::INDICES:
    ptr = lance_dataset_list_indices(dataset);
    break;
  default:
    ptr = nullptr;
    break;
  }

  vector<pair<string, string>> rows;
  try {
    rows = ParseTsvRows(ptr);
  } catch (...) {
    lance_close_dataset(dataset);
    throw;
  }

  lance_close_dataset(dataset);
  return make_uniq<LanceKvListGlobalState>(std::move(rows));
}

static void LanceKvListFunc(ClientContext &, TableFunctionInput &data,
                            DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceKvListGlobalState>();
  if (gstate.offset >= gstate.rows.size()) {
    output.SetCardinality(0);
    return;
  }

  auto remaining = gstate.rows.size() - gstate.offset;
  auto count = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);
  output.SetCardinality(count);

  for (idx_t i = 0; i < count; i++) {
    auto &kv = gstate.rows[gstate.offset + i];
    output.SetValue(0, i, Value(kv.first));
    output.SetValue(1, i, Value(kv.second));
  }
  gstate.offset += count;
}

static unique_ptr<FunctionData>
LanceKvListBind(ClientContext &context, TableFunctionBindInput &input,
                vector<LogicalType> &return_types, vector<string> &names,
                LanceKvTarget target, idx_t expected_inputs) {
  if (input.inputs.size() != expected_inputs) {
    throw BinderException("invalid argument count");
  }

  auto dataset_uri =
      ResolveLanceDatasetUriFromTableName(context, input.inputs[0]);

  string field_path;
  if (target == LanceKvTarget::FIELD_METADATA) {
    if (input.inputs[1].IsNull()) {
      throw BinderException("column name cannot be NULL");
    }
    field_path = input.inputs[1].GetValue<string>();
    if (field_path.empty()) {
      throw BinderException("column name cannot be empty");
    }
  }

  return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
  names = {"Key", "Value"};
  return make_uniq<LanceKvListBindData>(std::move(dataset_uri), target,
                                        std::move(field_path));
}

static unique_ptr<FunctionData>
LanceSetConfigBind(ClientContext &context, TableFunctionBindInput &input,
                   vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::CONFIG, 3);
}
static unique_ptr<FunctionData>
LanceUnsetConfigBind(ClientContext &context, TableFunctionBindInput &input,
                     vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::CONFIG, 2);
}
static unique_ptr<FunctionData>
LanceSetTableMetadataBind(ClientContext &context, TableFunctionBindInput &input,
                          vector<LogicalType> &return_types,
                          vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::TABLE_METADATA, 3);
}
static unique_ptr<FunctionData> LanceUnsetTableMetadataBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::TABLE_METADATA, 2);
}
static unique_ptr<FunctionData> LanceSetSchemaMetadataBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::SCHEMA_METADATA, 3);
}
static unique_ptr<FunctionData> LanceUnsetSchemaMetadataBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::SCHEMA_METADATA, 2);
}
static unique_ptr<FunctionData> LanceSetColumnMetadataBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::FIELD_METADATA, 4);
}
static unique_ptr<FunctionData> LanceUnsetColumnMetadataBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::FIELD_METADATA, 3);
}

static unique_ptr<FunctionData>
LanceConfigBind(ClientContext &context, TableFunctionBindInput &input,
                vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvListBind(context, input, return_types, names,
                         LanceKvTarget::CONFIG, 1);
}
static unique_ptr<FunctionData>
LanceTableMetadataBind(ClientContext &context, TableFunctionBindInput &input,
                       vector<LogicalType> &return_types,
                       vector<string> &names) {
  return LanceKvListBind(context, input, return_types, names,
                         LanceKvTarget::TABLE_METADATA, 1);
}
static unique_ptr<FunctionData>
LanceSchemaMetadataBind(ClientContext &context, TableFunctionBindInput &input,
                        vector<LogicalType> &return_types,
                        vector<string> &names) {
  return LanceKvListBind(context, input, return_types, names,
                         LanceKvTarget::SCHEMA_METADATA, 1);
}
static unique_ptr<FunctionData>
LanceColumnMetadataBind(ClientContext &context, TableFunctionBindInput &input,
                        vector<LogicalType> &return_types,
                        vector<string> &names) {
  return LanceKvListBind(context, input, return_types, names,
                         LanceKvTarget::FIELD_METADATA, 2);
}

static unique_ptr<FunctionData> LanceCreateScalarIndexBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvUpdateBind(context, input, return_types, names,
                           LanceKvTarget::INDICES, 3);
}
static unique_ptr<FunctionData>
LanceIndicesBind(ClientContext &context, TableFunctionBindInput &input,
                 vector<LogicalType> &return_types, vector<string> &names) {
  return LanceKvListBind(context, input, return_types, names,
                         LanceKvTarget::INDICES, 1);
}

struct LanceMaintenanceBindData final : public FunctionData {
  LanceMaintenanceBindData(string dataset_uri_p, int64_t older_than_seconds_p,
                           bool delete_unverified_p)
      : dataset_uri(std::move(dataset_uri_p)),
        older_than_seconds(older_than_seconds_p),
        delete_unverified(delete_unverified_p) {}

  string dataset_uri;
  int64_t older_than_seconds = 0;
  bool delete_unverified = false;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceMaintenanceBindData>(dataset_uri, older_than_seconds,
                                               delete_unverified);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceMaintenanceBindData>();
    return dataset_uri == other.dataset_uri &&
           older_than_seconds == other.older_than_seconds &&
           delete_unverified == other.delete_unverified;
  }
};

static unique_ptr<FunctionData>
LanceCompactFilesBind(ClientContext &context, TableFunctionBindInput &input,
                      vector<LogicalType> &return_types,
                      vector<string> &names) {
  if (input.inputs.size() != 1) {
    throw BinderException("lance_compact_files requires 1 argument");
  }
  auto dataset_uri =
      ResolveLanceDatasetUriFromTableName(context, input.inputs[0]);
  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  return make_uniq<LanceMaintenanceBindData>(std::move(dataset_uri), 0, false);
}

static unique_ptr<FunctionData> LanceCleanupOldVersionsBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() != 3) {
    throw BinderException("lance_cleanup_old_versions requires 3 arguments");
  }
  auto dataset_uri =
      ResolveLanceDatasetUriFromTableName(context, input.inputs[0]);

  int64_t older_than_seconds = 0;
  if (!input.inputs[1].IsNull()) {
    older_than_seconds =
        input.inputs[1].DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  }
  bool delete_unverified = false;
  if (!input.inputs[2].IsNull()) {
    delete_unverified =
        input.inputs[2].DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }
  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  return make_uniq<LanceMaintenanceBindData>(
      std::move(dataset_uri), older_than_seconds, delete_unverified);
}

static void LanceCompactFilesFunc(ClientContext &context,
                                  TableFunctionInput &data, DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceSingleRowGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceMaintenanceBindData>();
  void *dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + bind_data.dataset_uri +
                      LanceFormatErrorSuffix());
  }
  auto rc = lance_dataset_compact_files(dataset);
  lance_close_dataset(dataset);
  if (rc != 0) {
    throw IOException("Failed to compact Lance dataset: " +
                      bind_data.dataset_uri + LanceFormatErrorSuffix());
  }
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BIGINT(1));
}

static void LanceCleanupOldVersionsFunc(ClientContext &context,
                                        TableFunctionInput &data,
                                        DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceSingleRowGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceMaintenanceBindData>();
  void *dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + bind_data.dataset_uri +
                      LanceFormatErrorSuffix());
  }
  auto rc =
      lance_dataset_cleanup_old_versions(dataset, bind_data.older_than_seconds,
                                         bind_data.delete_unverified ? 1 : 0);
  lance_close_dataset(dataset);
  if (rc != 0) {
    throw IOException("Failed to cleanup old versions for Lance dataset: " +
                      bind_data.dataset_uri + LanceFormatErrorSuffix());
  }
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BIGINT(1));
}

void RegisterLanceMetadata(ExtensionLoader &loader) {
  loader.RegisterFunction(TableFunction(
      "lance_set_config",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceKvUpdateFunc, LanceSetConfigBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_unset_config", {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceKvUpdateFunc, LanceUnsetConfigBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_set_table_metadata",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceKvUpdateFunc, LanceSetTableMetadataBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_unset_table_metadata",
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, LanceKvUpdateFunc,
      LanceUnsetTableMetadataBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_set_schema_metadata",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceKvUpdateFunc, LanceSetSchemaMetadataBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_unset_schema_metadata",
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, LanceKvUpdateFunc,
      LanceUnsetSchemaMetadataBind, LanceSingleRowInitGlobal));

  loader.RegisterFunction(TableFunction(
      "lance_set_column_metadata",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::VARCHAR},
      LanceKvUpdateFunc, LanceSetColumnMetadataBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_unset_column_metadata",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceKvUpdateFunc, LanceUnsetColumnMetadataBind,
      LanceSingleRowInitGlobal));

  loader.RegisterFunction(TableFunction("lance_config", {LogicalType::VARCHAR},
                                        LanceKvListFunc, LanceConfigBind,
                                        LanceKvListInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_table_metadata", {LogicalType::VARCHAR}, LanceKvListFunc,
      LanceTableMetadataBind, LanceKvListInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_schema_metadata", {LogicalType::VARCHAR}, LanceKvListFunc,
      LanceSchemaMetadataBind, LanceKvListInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_column_metadata", {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceKvListFunc, LanceColumnMetadataBind, LanceKvListInitGlobal));

  loader.RegisterFunction(TableFunction(
      "lance_create_scalar_index",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceKvUpdateFunc, LanceCreateScalarIndexBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction("lance_indices", {LogicalType::VARCHAR},
                                        LanceKvListFunc, LanceIndicesBind,
                                        LanceKvListInitGlobal));

  loader.RegisterFunction(TableFunction(
      "lance_compact_files", {LogicalType::VARCHAR}, LanceCompactFilesFunc,
      LanceCompactFilesBind, LanceSingleRowInitGlobal));
  loader.RegisterFunction(TableFunction(
      "lance_cleanup_old_versions",
      {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BOOLEAN},
      LanceCleanupOldVersionsFunc, LanceCleanupOldVersionsBind,
      LanceSingleRowInitGlobal));
}

} // namespace duckdb
