#include "duckdb/storage/storage_extension.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/catalog/catalog_entry/copy_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/view_catalog_entry.hpp"
#include "duckdb/catalog/default/default_generator.hpp"
#include "duckdb/catalog/default/default_schemas.hpp"
#include "duckdb/catalog/duck_catalog.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception_format_value.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction.hpp"

#include "lance_common.hpp"
#include "lance_delete.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_table_entry.hpp"
#include "lance_update.hpp"

#include <cstring>

#include <algorithm>

namespace duckdb {

struct LanceDirectoryNamespaceConfig {
  string root;
  vector<string> option_keys;
  vector<string> option_values;
};

static string GetLanceNamespaceEndpoint(const AttachInfo &info) {
  for (auto &kv : info.options) {
    if (!StringUtil::CIEquals(kv.first, "endpoint") || kv.second.IsNull()) {
      continue;
    }
    auto endpoint =
        kv.second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
    if (!endpoint.empty()) {
      return endpoint;
    }
    break;
  }
  return "";
}

static string GetLanceNamespaceDelimiter(const AttachInfo &info) {
  for (auto &kv : info.options) {
    if (!StringUtil::CIEquals(kv.first, "delimiter") || kv.second.IsNull()) {
      continue;
    }
    auto delimiter =
        kv.second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
    return delimiter;
  }
  return "";
}

static void PopulateLanceTableColumnsFromDataset(ClientContext &context,
                                                 void *dataset,
                                                 ColumnList &out_columns) {
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

  for (idx_t i = 0; i < names.size(); i++) {
    out_columns.AddColumn(ColumnDefinition(names[i], types[i]));
  }
}

static string JoinNamespacePath(const string &root, const string &child) {
  if (root.empty()) {
    return child;
  }
  if (root.back() == '/' || root.back() == '\\') {
    return root + child;
  }
  return root + "/" + child;
}

static string GetDatasetDirName(const string &table_name);
static bool IsSafeDatasetTableName(const string &name);

static vector<string>
ListDirectoryNamespaceTables(const LanceDirectoryNamespaceConfig &ns) {
  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(ns.option_keys, ns.option_values, key_ptrs,
                                  value_ptrs);

  auto *ptr = lance_dir_namespace_list_tables(
      ns.root.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), ns.option_keys.size());
  if (!ptr) {
    throw IOException("Failed to list tables from Lance directory namespace: " +
                      ns.root + LanceFormatErrorSuffix());
  }
  string joined = ptr;
  lance_free_string(ptr);

  vector<string> out;
  for (auto &p : StringUtil::Split(joined, '\n')) {
    if (!p.empty()) {
      out.push_back(std::move(p));
    }
  }
  return out;
}

static vector<string> ListRestNamespaceTables(const string &endpoint,
                                              const string &namespace_id,
                                              const string &bearer_token,
                                              const string &api_key,
                                              const string &delimiter) {
  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();

  auto *ptr =
      lance_namespace_list_tables(endpoint.c_str(), namespace_id.c_str(),
                                  bearer_ptr, api_key_ptr, delimiter_ptr);
  if (!ptr) {
    throw IOException("Failed to list tables from Lance namespace: " +
                      endpoint + "/" + namespace_id + LanceFormatErrorSuffix());
  }
  string joined = ptr;
  lance_free_string(ptr);

  vector<string> out;
  for (auto &p : StringUtil::Split(joined, '\n')) {
    if (!p.empty()) {
      out.push_back(std::move(p));
    }
  }
  return out;
}

static bool
DirectoryNamespaceTableExists(const LanceDirectoryNamespaceConfig &ns,
                              const string &table_name) {
  auto tables = ListDirectoryNamespaceTables(ns);
  for (auto &t : tables) {
    if (StringUtil::CIEquals(t, table_name)) {
      return true;
    }
  }
  return false;
}

class LanceDirectoryDefaultGenerator : public DefaultGenerator {
public:
  LanceDirectoryDefaultGenerator(Catalog &catalog, SchemaCatalogEntry &schema,
                                 shared_ptr<LanceDirectoryNamespaceConfig> ns)
      : DefaultGenerator(catalog), schema(schema), ns(std::move(ns)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    if (!ns) {
      throw InternalException("Lance directory namespace config is missing");
    }
    if (!IsSafeDatasetTableName(entry_name)) {
      throw InvalidInputException(
          "Unsafe Lance dataset name for directory namespace: " + entry_name);
    }

    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(ns->option_keys, ns->option_values,
                                    key_ptrs, value_ptrs);

    const char *uri_ptr = nullptr;
    auto *dataset = lance_open_dataset_in_dir_namespace(
        ns->root.c_str(), entry_name.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(),
        ns->option_keys.size(), &uri_ptr);
    string dataset_uri;
    if (uri_ptr) {
      dataset_uri = uri_ptr;
      lance_free_string(uri_ptr);
    }
    if (!dataset) {
      return nullptr;
    }

    CreateTableInfo info(schema, entry_name);
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    try {
      PopulateLanceTableColumnsFromDataset(context, dataset, info.columns);
    } catch (...) {
      lance_close_dataset(dataset);
      throw;
    }
    lance_close_dataset(dataset);

    if (dataset_uri.empty()) {
      dataset_uri = JoinNamespacePath(ns->root, GetDatasetDirName(entry_name));
    }
    return make_uniq_base<CatalogEntry, LanceTableEntry>(
        catalog, schema, info, std::move(dataset_uri));
  }

  vector<string> GetDefaultEntries() override {
    if (!ns) {
      return {};
    }
    return ListDirectoryNamespaceTables(*ns);
  }

private:
  SchemaCatalogEntry &schema;
  shared_ptr<LanceDirectoryNamespaceConfig> ns;
};

class LanceRestNamespaceDefaultGenerator : public DefaultGenerator {
public:
  LanceRestNamespaceDefaultGenerator(
      Catalog &catalog, SchemaCatalogEntry &schema, string endpoint,
      string namespace_id, string bearer_token, string api_key,
      string delimiter, string bearer_token_override, string api_key_override)
      : DefaultGenerator(catalog), schema(schema),
        endpoint(std::move(endpoint)), namespace_id(std::move(namespace_id)),
        bearer_token(std::move(bearer_token)), api_key(std::move(api_key)),
        delimiter(std::move(delimiter)),
        bearer_token_override(std::move(bearer_token_override)),
        api_key_override(std::move(api_key_override)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    string table_uri;
    auto *dataset =
        LanceOpenDatasetInNamespace(context, endpoint, entry_name, bearer_token,
                                    api_key, delimiter, table_uri);
    if (!dataset) {
      return nullptr;
    }
    lance_close_dataset(dataset);

    auto view = make_uniq<CreateViewInfo>();
    view->schema = DEFAULT_SCHEMA;
    view->view_name = entry_name;
    auto view_sql = StringUtil::Format(
        "SELECT * FROM __lance_namespace_scan(%s, %s, %s", SQLString(endpoint),
        SQLString(entry_name), SQLString(delimiter));
    if (!bearer_token_override.empty()) {
      view_sql += StringUtil::Format(", bearer_token=%s",
                                     SQLString(bearer_token_override));
    }
    if (!api_key_override.empty()) {
      view_sql +=
          StringUtil::Format(", api_key=%s", SQLString(api_key_override));
    }
    view->sql = view_sql + ")";
    view->internal = false;
    view->on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    auto view_info = CreateViewInfo::FromSelect(context, std::move(view));
    return make_uniq_base<CatalogEntry, ViewCatalogEntry>(catalog, schema,
                                                          *view_info);
  }

  vector<string> GetDefaultEntries() override {
    return ListRestNamespaceTables(endpoint, namespace_id, bearer_token,
                                   api_key, delimiter);
  }

private:
  SchemaCatalogEntry &schema;
  string endpoint;
  string namespace_id;
  string bearer_token;
  string api_key;
  string delimiter;
  string bearer_token_override;
  string api_key_override;
};

static string GetDatasetDirName(const string &table_name) {
  return table_name + ".lance";
}

static bool IsSafeDatasetTableName(const string &name) {
  if (name.empty()) {
    return false;
  }
  if (name == "." || name == "..") {
    return false;
  }
  if (name.find('/') != string::npos || name.find('\\') != string::npos) {
    return false;
  }
  return true;
}

static string CreateTableModeFromConflict(OnCreateConflict on_conflict) {
  switch (on_conflict) {
  case OnCreateConflict::ERROR_ON_CONFLICT:
  case OnCreateConflict::IGNORE_ON_CONFLICT:
    return "create";
  case OnCreateConflict::REPLACE_ON_CONFLICT:
    return "overwrite";
  case OnCreateConflict::ALTER_ON_CONFLICT:
    break;
  default:
    break;
  }
  return "overwrite";
}

class LanceSchemaEntry final : public DuckSchemaEntry {
public:
  LanceSchemaEntry(Catalog &catalog, CreateSchemaInfo &info,
                   shared_ptr<LanceDirectoryNamespaceConfig> directory_ns,
                   bool is_rest_namespace)
      : DuckSchemaEntry(catalog, info), directory_ns(std::move(directory_ns)),
        is_rest_namespace(is_rest_namespace) {}

  void DropEntry(ClientContext &context, DropInfo &info) override {
    if (info.type != CatalogType::TABLE_ENTRY) {
      DuckSchemaEntry::DropEntry(context, info);
      return;
    }
    if (is_rest_namespace) {
      throw NotImplementedException(
          "DROP TABLE is not supported for Lance REST namespaces");
    }

    // DuckDB stores TABLE and VIEW entries in the same catalog set (see
    // DuckSchemaEntry::GetCatalogSet), so we need to resolve the existing entry
    // type explicitly before attempting to drop anything.
    auto transaction = GetCatalogTransaction(context);
    auto &set = GetCatalogSet(info.type);
    auto existing_entry = set.GetEntry(transaction, info.name);
    if (!existing_entry) {
      throw InternalException(
          "Failed to drop entry \"%s\" - entry could not be found", info.name);
    }
    auto existing_type = existing_entry->type;

    if (!IsSafeDatasetTableName(info.name)) {
      throw InvalidInputException("Unsafe Lance dataset name for DROP TABLE: " +
                                  info.name);
    }

    if (!directory_ns || directory_ns->root.empty()) {
      throw InternalException("Lance directory namespace root is empty");
    }
    auto root = directory_ns->root;

    vector<string> option_keys;
    vector<string> option_values;
    option_keys = directory_ns->option_keys;
    option_values = directory_ns->option_values;

    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                    value_ptrs);

    auto rc = lance_dir_namespace_drop_table(
        root.c_str(), info.name.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size());
    if (rc != 0) {
      throw IOException("Failed to drop Lance dataset: " + root + "/" +
                        GetDatasetDirName(info.name) +
                        LanceFormatErrorSuffix());
    }

    // Drop the DuckDB catalog entry after the dataset has been deleted
    // successfully.
    //
    // Note: TABLE_ENTRY and VIEW_ENTRY share the same underlying catalog set.
    // DuckDB's transactional DROP path assumes TABLE_ENTRY is always a
    // DuckTableEntry (and will fail for extension-backed TableCatalogEntry
    // implementations). To avoid that, perform the catalog drop using a system
    // (non-transactional) CatalogTransaction.
    if (existing_type != CatalogType::TABLE_ENTRY &&
        existing_type != CatalogType::VIEW_ENTRY) {
      throw InternalException(
          "Unexpected catalog entry type for DROP TABLE: %s",
          CatalogTypeToString(existing_type));
    }
    auto system_transaction =
        CatalogTransaction::GetSystemTransaction(catalog.GetDatabase());
    if (!set.DropEntry(system_transaction, info.name, info.cascade, true)) {
      throw InternalException(
          "Could not drop element because of an internal error");
    }
  }

  optional_ptr<CatalogEntry> CreateTable(CatalogTransaction transaction,
                                         BoundCreateTableInfo &info) override {
    auto &create_info = info.Base();
    if (create_info.temporary) {
      throw NotImplementedException(
          "Lance ATTACH TYPE LANCE does not support TEMPORARY tables");
    }
    if (is_rest_namespace) {
      throw NotImplementedException(
          "CREATE TABLE is not supported for Lance REST namespaces");
    }
    if (!info.constraints.empty() || !create_info.constraints.empty()) {
      throw NotImplementedException(
          "Lance CREATE TABLE does not support constraints");
    }
    if (!IsSafeDatasetTableName(create_info.table)) {
      throw InvalidInputException(
          "Unsafe Lance dataset name for CREATE TABLE: " + create_info.table);
    }
    if (!directory_ns || directory_ns->root.empty()) {
      throw InternalException("Lance directory namespace root is empty");
    }

    auto &context = transaction.GetContext();
    auto dataset_path = JoinNamespacePath(directory_ns->root,
                                          GetDatasetDirName(create_info.table));

    auto exists =
        DirectoryNamespaceTableExists(*directory_ns, create_info.table);
    if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
        exists) {
      return nullptr;
    }
    if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
        exists) {
      throw IOException("Lance dataset already exists: " + dataset_path);
    }

    vector<string> names;
    vector<LogicalType> types;
    names.reserve(create_info.columns.LogicalColumnCount());
    types.reserve(create_info.columns.LogicalColumnCount());
    for (auto &col : create_info.columns.Logical()) {
      names.push_back(col.Name());
      types.push_back(col.Type());
    }

    ArrowSchemaWrapper schema_root;
    memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
    auto props = context.GetClientProperties();
    ArrowConverter::ToArrowSchema(&schema_root.arrow_schema, types, names,
                                  props);

    auto mode = CreateTableModeFromConflict(create_info.on_conflict);

    vector<string> option_keys;
    vector<string> option_values;
    option_keys = directory_ns->option_keys;
    option_values = directory_ns->option_values;
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                    value_ptrs);

    auto *writer = lance_open_writer_with_storage_options(
        dataset_path.c_str(), mode.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
        LANCE_DEFAULT_MAX_ROWS_PER_FILE, LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
        LANCE_DEFAULT_MAX_BYTES_PER_FILE, &schema_root.arrow_schema);
    if (!writer) {
      throw IOException("Failed to open Lance writer: " + dataset_path +
                        LanceFormatErrorSuffix());
    }
    auto rc = lance_writer_finish(writer);
    lance_close_writer(writer);
    if (rc != 0) {
      throw IOException("Failed to finalize Lance dataset write" +
                        LanceFormatErrorSuffix());
    }

    return nullptr;
  }

private:
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  bool is_rest_namespace;
};

class LanceDuckCatalog final : public DuckCatalog {
public:
  using DuckCatalog::PlanDelete;

  LanceDuckCatalog(AttachedDatabase &db,
                   shared_ptr<LanceDirectoryNamespaceConfig> directory_ns,
                   bool is_rest_namespace)
      : DuckCatalog(db), directory_ns(std::move(directory_ns)),
        is_rest_namespace(is_rest_namespace) {}

  using DuckCatalog::PlanUpdate;

  PhysicalOperator &PlanUpdate(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalUpdate &op) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceUpdateOverwrite(context, planner, op);
    }
    return Catalog::PlanUpdate(context, planner, op);
  }

  PhysicalOperator &PlanInsert(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalInsert &op,
                               optional_ptr<PhysicalOperator> plan) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceInsertAppend(context, planner, op, plan);
    }
    return DuckCatalog::PlanInsert(context, planner, op, plan);
  }

  PhysicalOperator &PlanDelete(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalDelete &op,
                               PhysicalOperator &plan) override;

  PhysicalOperator &PlanCreateTableAs(ClientContext &context,
                                      PhysicalPlanGenerator &planner,
                                      LogicalCreateTable &op,
                                      PhysicalOperator &plan) override {
    auto &create_info = op.info->Base();
    if (create_info.temporary) {
      throw NotImplementedException(
          "Lance ATTACH TYPE LANCE does not support TEMPORARY tables");
    }
    if (is_rest_namespace) {
      throw NotImplementedException(
          "CREATE TABLE is not supported for Lance REST namespaces");
    }
    if (!IsSafeDatasetTableName(create_info.table)) {
      throw InvalidInputException(
          "Unsafe Lance dataset name for CREATE TABLE: " + create_info.table);
    }
    if (!directory_ns || directory_ns->root.empty()) {
      throw InternalException("Lance directory namespace root is empty");
    }

    auto dataset_path = JoinNamespacePath(directory_ns->root,
                                          GetDatasetDirName(create_info.table));

    auto exists =
        DirectoryNamespaceTableExists(*directory_ns, create_info.table);

    if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
        exists) {
      return planner.Make<PhysicalEmptyResult>(op.types,
                                               op.estimated_cardinality);
    }
    if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
        exists) {
      throw IOException("Lance dataset already exists: " + dataset_path);
    }

    auto mode = CreateTableModeFromConflict(create_info.on_conflict);

    CopyInfo copy_info;
    copy_info.is_from = false;
    copy_info.format = "lance";
    copy_info.file_path = dataset_path;
    copy_info.options["mode"] = {Value(mode)};

    auto &system_catalog = Catalog::GetSystemCatalog(context);
    auto entry = system_catalog.GetEntry(
        context, CatalogType::COPY_FUNCTION_ENTRY, DEFAULT_SCHEMA, "lance",
        OnEntryNotFound::THROW_EXCEPTION);
    auto &copy_function = entry->Cast<CopyFunctionCatalogEntry>().function;

    if (!copy_function.copy_to_bind) {
      throw NotImplementedException(
          "COPY TO is not supported for FORMAT \"lance\"");
    }

    auto names = create_info.columns.GetColumnNames();
    auto types = create_info.columns.GetColumnTypes();
    CopyFunctionBindInput bind_input(copy_info);
    auto bind_data =
        copy_function.copy_to_bind(context, bind_input, names, types);

    bool preserve_insertion_order =
        PhysicalPlanGenerator::PreserveInsertionOrder(context, plan);
    bool supports_batch_index =
        PhysicalPlanGenerator::UseBatchIndex(context, plan);
    auto execution_mode = CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
    if (copy_function.execution_mode) {
      execution_mode = copy_function.execution_mode(preserve_insertion_order,
                                                    supports_batch_index);
    }

    if (execution_mode == CopyFunctionExecutionMode::BATCH_COPY_TO_FILE) {
      auto &copy = planner.Make<PhysicalBatchCopyToFile>(
          op.types, copy_function, std::move(bind_data),
          op.estimated_cardinality);
      auto &cast_copy = copy.Cast<PhysicalBatchCopyToFile>();
      cast_copy.file_path = dataset_path;
      cast_copy.use_tmp_file = false;
      cast_copy.return_type = CopyFunctionReturnType::CHANGED_ROWS;
      cast_copy.write_empty_file = true;
      cast_copy.children.push_back(plan);
      return copy;
    }

    auto &copy = planner.Make<PhysicalCopyToFile>(op.types, copy_function,
                                                  std::move(bind_data),
                                                  op.estimated_cardinality);
    auto &cast_copy = copy.Cast<PhysicalCopyToFile>();
    cast_copy.file_path = dataset_path;
    cast_copy.use_tmp_file = false;
    cast_copy.filename_pattern = FilenamePattern();
    cast_copy.file_extension = "";
    cast_copy.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;
    cast_copy.return_type = CopyFunctionReturnType::CHANGED_ROWS;
    cast_copy.per_thread_output = false;
    cast_copy.file_size_bytes = optional_idx();
    cast_copy.rotate = false;
    cast_copy.write_empty_file = true;
    cast_copy.partition_output = false;
    cast_copy.write_partition_columns = false;
    cast_copy.hive_file_pattern = false;
    cast_copy.partition_columns.clear();
    cast_copy.names = names;
    cast_copy.expected_types = types;
    cast_copy.parallel =
        execution_mode == CopyFunctionExecutionMode::PARALLEL_COPY_TO_FILE;
    cast_copy.children.push_back(plan);
    return copy;
  }

  PhysicalOperator &PlanDelete(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalDelete &op) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceDelete(context, planner, op);
    }
    return Catalog::PlanDelete(context, planner, op);
  }

  void ReplaceDefaultSchemaWithLanceSchema(CatalogTransaction transaction) {
    auto &schemas = GetSchemaCatalogSet();
    (void)schemas.DropEntry(transaction, DEFAULT_SCHEMA, true, true);

    CreateSchemaInfo info;
    info.schema = DEFAULT_SCHEMA;
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    LogicalDependencyList dependencies;
    auto entry = make_uniq<LanceSchemaEntry>(*this, info, directory_ns,
                                             is_rest_namespace);
    if (!schemas.CreateEntry(transaction, info.schema, std::move(entry),
                             dependencies)) {
      throw InternalException("Failed to replace Lance schema entry");
    }
  }

private:
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  bool is_rest_namespace;
};

PhysicalOperator &LanceDuckCatalog::PlanDelete(ClientContext &context,
                                               PhysicalPlanGenerator &planner,
                                               LogicalDelete &op,
                                               PhysicalOperator &plan) {
  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    return DuckCatalog::PlanDelete(context, planner, op, plan);
  }
  (void)plan;
  return PlanLanceDelete(context, planner, op);
}

static unique_ptr<Catalog>
LanceStorageAttach(optional_ptr<StorageExtensionInfo>, ClientContext &context,
                   AttachedDatabase &db, const string &name, AttachInfo &info,
                   AttachOptions &) {
  auto attach_path = info.path;
  auto endpoint = GetLanceNamespaceEndpoint(info);
  auto delimiter = GetLanceNamespaceDelimiter(info);

  unique_ptr<DefaultGenerator> generator;
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;

  auto is_rest_namespace = !endpoint.empty();
  string namespace_id;
  string bearer_token;
  string api_key;
  string bearer_token_override;
  string api_key_override;

  if (!is_rest_namespace) {
    auto root = FileSystem::GetFileSystem(context).ExpandPath(attach_path);
    vector<string> option_keys;
    vector<string> option_values;
    string open_root;
    ResolveLanceStorageOptions(context, root, open_root, option_keys,
                               option_values);

    string list_error;
    vector<string> discovered_tables;
    // Validate the namespace during ATTACH.
    if (!TryLanceDirNamespaceListTables(context, open_root, discovered_tables,
                                        list_error)) {
      throw IOException(
          "Failed to list tables from Lance directory namespace: " +
          list_error);
    }
    directory_ns = make_shared_ptr<LanceDirectoryNamespaceConfig>();
    directory_ns->root = std::move(open_root);
    directory_ns->option_keys = std::move(option_keys);
    directory_ns->option_values = std::move(option_values);
  } else {
    namespace_id = attach_path;
    if (namespace_id.empty()) {
      throw InvalidInputException(
          "ATTACH TYPE LANCE with ENDPOINT requires a non-empty namespace id");
    }
    ResolveLanceNamespaceAuth(context, endpoint, info.options, bearer_token,
                              api_key);
    ResolveLanceNamespaceAuthOverrides(info.options, bearer_token_override,
                                       api_key_override);
    string list_error;
    vector<string> discovered_tables;
    // Validate the namespace during ATTACH.
    if (!TryLanceNamespaceListTables(context, endpoint, namespace_id,
                                     bearer_token, api_key, delimiter,
                                     discovered_tables, list_error)) {
      throw IOException("Failed to list tables from Lance namespace: " +
                        list_error);
    }
  }

  // Back the attached catalog by an in-memory DuckCatalog that lazily
  // materializes per-table entries mapping to `lance_scan` / internal namespace
  // scan, and supports CREATE TABLE for directory namespaces.
  info.path = ":memory:";
  auto catalog =
      make_uniq<LanceDuckCatalog>(db, directory_ns, is_rest_namespace);
  catalog->Initialize(false);

  auto system_transaction =
      CatalogTransaction::GetSystemTransaction(db.GetDatabase());
  catalog->ReplaceDefaultSchemaWithLanceSchema(system_transaction);
  auto &schema = catalog->GetSchema(system_transaction, DEFAULT_SCHEMA);

  auto &duck_schema = schema.Cast<DuckSchemaEntry>();
  auto &catalog_set = duck_schema.GetCatalogSet(
      is_rest_namespace ? CatalogType::VIEW_ENTRY : CatalogType::TABLE_ENTRY);

  if (!is_rest_namespace) {
    generator = make_uniq<LanceDirectoryDefaultGenerator>(*catalog, schema,
                                                          directory_ns);
  } else {
    generator = make_uniq<LanceRestNamespaceDefaultGenerator>(
        *catalog, schema, std::move(endpoint), std::move(namespace_id),
        std::move(bearer_token), std::move(api_key), std::move(delimiter),
        std::move(bearer_token_override), std::move(api_key_override));
  }
  catalog_set.SetDefaultGenerator(std::move(generator));

  (void)name;
  return std::move(catalog);
}

struct LancePendingAppend {
  string path;
  vector<string> option_keys;
  vector<string> option_values;
  void *transaction = nullptr;
};

class LanceTransactionManager final : public DuckTransactionManager {
public:
  explicit LanceTransactionManager(AttachedDatabase &db)
      : DuckTransactionManager(db) {}

  void RegisterPendingAppend(Transaction &transaction_p,
                             LancePendingAppend pending) {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    lock_guard<mutex> guard(pending_lock);
    pending_appends[transaction.transaction_id].push_back(std::move(pending));
  }

  ErrorData CommitTransaction(ClientContext &context,
                              Transaction &transaction_p) override {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    vector<LancePendingAppend> appends;
    {
      lock_guard<mutex> guard(pending_lock);
      auto it = pending_appends.find(transaction.transaction_id);
      if (it != pending_appends.end()) {
        appends = std::move(it->second);
        pending_appends.erase(it);
      }
    }

    for (idx_t i = 0; i < appends.size(); i++) {
      auto &pending = appends[i];
      vector<const char *> key_ptrs;
      vector<const char *> value_ptrs;
      BuildStorageOptionPointerArrays(
          pending.option_keys, pending.option_values, key_ptrs, value_ptrs);

      auto rc = lance_commit_transaction_with_storage_options(
          pending.path.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(),
          pending.option_keys.size(), pending.transaction);
      if (rc != 0) {
        // Best-effort cleanup of any remaining pending transactions.
        // Note: the transaction pointer is consumed by the commit call, even on
        // error.
        for (idx_t k = i + 1; k < appends.size(); k++) {
          lance_free_transaction(appends[k].transaction);
        }
        DuckTransactionManager::RollbackTransaction(transaction_p);
        return ErrorData(ExceptionType::TRANSACTION,
                         "Failed to commit Lance append transaction for '" +
                             pending.path + "'" + LanceFormatErrorSuffix());
      }
    }

    return DuckTransactionManager::CommitTransaction(context, transaction_p);
  }

  void RollbackTransaction(Transaction &transaction_p) override {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    vector<LancePendingAppend> appends;
    {
      lock_guard<mutex> guard(pending_lock);
      auto it = pending_appends.find(transaction.transaction_id);
      if (it != pending_appends.end()) {
        appends = std::move(it->second);
        pending_appends.erase(it);
      }
    }
    for (auto &pending : appends) {
      lance_free_transaction(pending.transaction);
    }
    DuckTransactionManager::RollbackTransaction(transaction_p);
  }

private:
  mutex pending_lock;
  unordered_map<transaction_t, vector<LancePendingAppend>> pending_appends;
};

static unique_ptr<TransactionManager>
LanceStorageTransactionManager(optional_ptr<StorageExtensionInfo>,
                               AttachedDatabase &db, Catalog &) {
  return make_uniq<LanceTransactionManager>(db);
}

void RegisterLanceStorage(DBConfig &config) {
  auto ext = make_uniq<StorageExtension>();
  ext->attach = LanceStorageAttach;
  ext->create_transaction_manager = LanceStorageTransactionManager;
  config.storage_extensions["lance"] = std::move(ext);
}

void RegisterLancePendingAppend(ClientContext &context, Catalog &catalog,
                                string dataset_uri, vector<string> option_keys,
                                vector<string> option_values,
                                void *lance_transaction) {
  auto &txn = Transaction::Get(context, catalog);
  auto *tm = dynamic_cast<LanceTransactionManager *>(&txn.manager);
  if (!tm) {
    lance_free_transaction(lance_transaction);
    throw InternalException(
        "RegisterLancePendingAppend requires LanceTransactionManager");
  }
  LancePendingAppend pending;
  pending.path = std::move(dataset_uri);
  pending.option_keys = std::move(option_keys);
  pending.option_values = std::move(option_values);
  pending.transaction = lance_transaction;
  tm->RegisterPendingAppend(txn, std::move(pending));
}

} // namespace duckdb
