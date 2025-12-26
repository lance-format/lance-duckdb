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
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"

#include <algorithm>

namespace duckdb {

struct LanceDirectoryTableList {
  explicit LanceDirectoryTableList(vector<string> init_tables)
      : tables(std::move(init_tables)) {}

  mutex lock;
  vector<string> tables;
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

class LanceDirectoryDefaultGenerator : public DefaultGenerator {
public:
  LanceDirectoryDefaultGenerator(Catalog &catalog, SchemaCatalogEntry &schema,
                                 string namespace_root,
                                 shared_ptr<LanceDirectoryTableList> table_list)
      : DefaultGenerator(catalog), schema(schema),
        namespace_root(std::move(namespace_root)),
        table_list(std::move(table_list)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    auto &fs = FileSystem::GetFileSystem(context);
    auto dataset_dir = entry_name + ".lance";
    auto dataset_path = fs.JoinPath(namespace_root, dataset_dir);

    bool is_known = false;
    if (table_list) {
      lock_guard<mutex> guard(table_list->lock);
      for (auto &t : table_list->tables) {
        if (StringUtil::CIEquals(t, entry_name)) {
          is_known = true;
          break;
        }
      }
    }

    if (!is_known &&
        !(fs.DirectoryExists(dataset_path) || fs.FileExists(dataset_path))) {
      return nullptr;
    }

    auto view = make_uniq<CreateViewInfo>();
    view->schema = DEFAULT_SCHEMA;
    view->view_name = entry_name;
    view->sql = StringUtil::Format("SELECT * FROM lance_scan(%s)",
                                   SQLString(dataset_path));
    view->internal = false;
    view->on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    auto view_info = CreateViewInfo::FromSelect(context, std::move(view));
    return make_uniq_base<CatalogEntry, ViewCatalogEntry>(catalog, schema,
                                                          *view_info);
  }

  vector<string> GetDefaultEntries() override {
    if (!table_list) {
      return {};
    }
    lock_guard<mutex> guard(table_list->lock);
    return table_list->tables;
  }

private:
  SchemaCatalogEntry &schema;
  string namespace_root;
  shared_ptr<LanceDirectoryTableList> table_list;
};

class LanceRestNamespaceDefaultGenerator : public DefaultGenerator {
public:
  LanceRestNamespaceDefaultGenerator(Catalog &catalog,
                                     SchemaCatalogEntry &schema,
                                     string endpoint, string delimiter,
                                     vector<string> discovered_tables)
      : DefaultGenerator(catalog), schema(schema),
        endpoint(std::move(endpoint)), delimiter(std::move(delimiter)),
        discovered_tables(std::move(discovered_tables)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    bool is_known = false;
    for (auto &t : discovered_tables) {
      if (StringUtil::CIEquals(t, entry_name)) {
        is_known = true;
        break;
      }
    }
    if (!is_known) {
      return nullptr;
    }

    auto view = make_uniq<CreateViewInfo>();
    view->schema = DEFAULT_SCHEMA;
    view->view_name = entry_name;
    view->sql = StringUtil::Format(
        "SELECT * FROM __lance_namespace_scan(%s, %s, %s)", SQLString(endpoint),
        SQLString(entry_name), SQLString(delimiter));
    view->internal = false;
    view->on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    auto view_info = CreateViewInfo::FromSelect(context, std::move(view));
    return make_uniq_base<CatalogEntry, ViewCatalogEntry>(catalog, schema,
                                                          *view_info);
  }

  vector<string> GetDefaultEntries() override { return discovered_tables; }

private:
  SchemaCatalogEntry &schema;
  string endpoint;
  string delimiter;
  vector<string> discovered_tables;
};

static constexpr uint64_t DEFAULT_MAX_ROWS_PER_FILE = 1024ULL * 1024ULL;
static constexpr uint64_t DEFAULT_MAX_ROWS_PER_GROUP = 1024ULL;
static constexpr uint64_t DEFAULT_MAX_BYTES_PER_FILE =
    90ULL * 1024ULL * 1024ULL * 1024ULL;

static string GetDatasetDirName(const string &table_name) {
  return table_name + ".lance";
}

static bool IsSafeDatasetTableName(const string &name) {
  if (name.empty()) {
    return false;
  }
  if (name.find('\\') != string::npos) {
    return false;
  }

  idx_t start = 0;
  while (start <= name.size()) {
    auto end = name.find('/', start);
    if (end == string::npos) {
      end = name.size();
    }
    auto part = name.substr(start, end - start);
    if (part.empty() || part == "." || part == "..") {
      return false;
    }
    if (end == name.size()) {
      break;
    }
    start = end + 1;
  }
  if (name.find("..") != string::npos) {
    return false;
  }
  return true;
}

static string CreateTableModeFromConflict(OnCreateConflict on_conflict) {
  switch (on_conflict) {
  case OnCreateConflict::ERROR_ON_CONFLICT:
  case OnCreateConflict::IGNORE_ON_CONFLICT:
  case OnCreateConflict::REPLACE_ON_CONFLICT:
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
                   string directory_namespace_root, bool is_rest_namespace,
                   shared_ptr<LanceDirectoryTableList> directory_table_list)
      : DuckSchemaEntry(catalog, info),
        directory_namespace_root(std::move(directory_namespace_root)),
        is_rest_namespace(is_rest_namespace),
        directory_table_list(std::move(directory_table_list)) {}

  void DropEntry(ClientContext &context, DropInfo &info) override {
    if (info.type != CatalogType::TABLE_ENTRY) {
      DuckSchemaEntry::DropEntry(context, info);
      return;
    }
    if (is_rest_namespace) {
      throw NotImplementedException(
          "DROP TABLE is not supported for Lance REST namespaces");
    }

    if (!IsSafeDatasetTableName(info.name)) {
      throw InvalidInputException("Unsafe Lance dataset name for DROP TABLE: " +
                                  info.name);
    }

    auto &fs = FileSystem::GetFileSystem(context);
    auto root = LanceNormalizeS3Scheme(directory_namespace_root);
    if (root.empty()) {
      throw InternalException("Lance directory namespace root is empty");
    }
    auto dataset_path = fs.JoinPath(root, GetDatasetDirName(info.name));
    dataset_path = LanceNormalizeS3Scheme(dataset_path);

    auto &views = GetCatalogSet(CatalogType::VIEW_ENTRY);
    auto view_entry = views.GetEntry(context, info.name);
    const bool view_exists = view_entry != nullptr;

    const bool dir_exists = fs.DirectoryExists(dataset_path);
    const bool file_exists = fs.FileExists(dataset_path);
    const bool dataset_exists = dir_exists || file_exists;

    if (!view_exists && !dataset_exists) {
      DuckSchemaEntry::DropEntry(context, info);
      return;
    }

    // Prefer dropping the view first to preserve normal DROP semantics for
    // dependent objects (unless CASCADE is used).
    DropInfo view_drop(info);
    view_drop.type = CatalogType::VIEW_ENTRY;
    view_drop.if_not_found = OnEntryNotFound::RETURN_NULL;
    view_drop.allow_drop_internal = true;
    DuckSchemaEntry::DropEntry(context, view_drop);

    if (dir_exists) {
      fs.RemoveDirectory(dataset_path);
    } else if (file_exists) {
      fs.RemoveFile(dataset_path);
    }

    if (directory_table_list) {
      lock_guard<mutex> guard(directory_table_list->lock);
      auto &tables = directory_table_list->tables;
      tables.erase(remove_if(tables.begin(), tables.end(),
                             [&](const string &t) {
                               return StringUtil::CIEquals(t, info.name);
                             }),
                   tables.end());
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

    auto &context = transaction.GetContext();
    auto &fs = FileSystem::GetFileSystem(context);
    auto dataset_path = fs.JoinPath(directory_namespace_root,
                                    GetDatasetDirName(create_info.table));

    auto exists =
        fs.DirectoryExists(dataset_path) || fs.FileExists(dataset_path);
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
    auto *writer = lance_open_writer_with_storage_options(
        dataset_path.c_str(), mode.c_str(), nullptr, nullptr, 0,
        DEFAULT_MAX_ROWS_PER_FILE, DEFAULT_MAX_ROWS_PER_GROUP,
        DEFAULT_MAX_BYTES_PER_FILE, &schema_root.arrow_schema);
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
  string directory_namespace_root;
  bool is_rest_namespace;
  shared_ptr<LanceDirectoryTableList> directory_table_list;
};

class LanceDuckCatalog final : public DuckCatalog {
public:
  LanceDuckCatalog(AttachedDatabase &db, string directory_namespace_root,
                   bool is_rest_namespace,
                   shared_ptr<LanceDirectoryTableList> directory_table_list)
      : DuckCatalog(db),
        directory_namespace_root(std::move(directory_namespace_root)),
        is_rest_namespace(is_rest_namespace),
        directory_table_list(std::move(directory_table_list)) {}

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

    auto &fs = FileSystem::GetFileSystem(context);
    auto dataset_path = fs.JoinPath(directory_namespace_root,
                                    GetDatasetDirName(create_info.table));

    auto exists =
        fs.DirectoryExists(dataset_path) || fs.FileExists(dataset_path);

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

  void ReplaceDefaultSchemaWithLanceSchema(CatalogTransaction transaction) {
    auto &schemas = GetSchemaCatalogSet();
    (void)schemas.DropEntry(transaction, DEFAULT_SCHEMA, true, true);

    CreateSchemaInfo info;
    info.schema = DEFAULT_SCHEMA;
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    LogicalDependencyList dependencies;
    auto entry =
        make_uniq<LanceSchemaEntry>(*this, info, directory_namespace_root,
                                    is_rest_namespace, directory_table_list);
    if (!schemas.CreateEntry(transaction, info.schema, std::move(entry),
                             dependencies)) {
      throw InternalException("Failed to replace Lance schema entry");
    }
  }

private:
  string directory_namespace_root;
  bool is_rest_namespace;
  shared_ptr<LanceDirectoryTableList> directory_table_list;
};

static unique_ptr<Catalog>
LanceStorageAttach(optional_ptr<StorageExtensionInfo>, ClientContext &context,
                   AttachedDatabase &db, const string &name, AttachInfo &info,
                   AttachOptions &) {
  auto attach_path = info.path;
  auto endpoint = GetLanceNamespaceEndpoint(info);
  auto delimiter = GetLanceNamespaceDelimiter(info);

  unique_ptr<DefaultGenerator> generator;
  vector<string> discovered_tables;
  shared_ptr<LanceDirectoryTableList> directory_table_list;

  auto is_rest_namespace = !endpoint.empty();
  string directory_namespace_root;

  if (!is_rest_namespace) {
    directory_namespace_root =
        FileSystem::GetFileSystem(context).ExpandPath(attach_path);
    string list_error;
    if (!TryLanceDirNamespaceListTables(context, directory_namespace_root,
                                        discovered_tables, list_error)) {
      throw IOException(
          "Failed to list tables from Lance directory namespace: " +
          list_error);
    }
    directory_table_list =
        make_shared_ptr<LanceDirectoryTableList>(std::move(discovered_tables));
  } else {
    const auto namespace_id = attach_path;
    if (namespace_id.empty()) {
      throw InvalidInputException(
          "ATTACH TYPE LANCE with ENDPOINT requires a non-empty namespace id");
    }
    string bearer_token;
    string api_key;
    ResolveLanceNamespaceAuth(context, endpoint, info.options, bearer_token,
                              api_key);
    string list_error;
    if (!TryLanceNamespaceListTables(context, endpoint, namespace_id,
                                     bearer_token, api_key, delimiter,
                                     discovered_tables, list_error)) {
      throw IOException("Failed to list tables from Lance namespace: " +
                        list_error);
    }
  }

  // Back the attached catalog by an in-memory DuckCatalog that lazily
  // materializes per-table views mapping to `lance_scan` / internal namespace
  // scan, and supports CREATE TABLE for directory namespaces.
  info.path = ":memory:";
  auto catalog = make_uniq<LanceDuckCatalog>(
      db, directory_namespace_root, is_rest_namespace, directory_table_list);
  catalog->Initialize(false);

  auto system_transaction =
      CatalogTransaction::GetSystemTransaction(db.GetDatabase());
  catalog->ReplaceDefaultSchemaWithLanceSchema(system_transaction);
  auto &schema = catalog->GetSchema(system_transaction, DEFAULT_SCHEMA);

  auto &duck_schema = schema.Cast<DuckSchemaEntry>();
  auto &catalog_set = duck_schema.GetCatalogSet(CatalogType::VIEW_ENTRY);

  if (!is_rest_namespace) {
    generator = make_uniq<LanceDirectoryDefaultGenerator>(
        *catalog, schema, std::move(directory_namespace_root),
        directory_table_list);
  } else {
    generator = make_uniq<LanceRestNamespaceDefaultGenerator>(
        *catalog, schema, std::move(endpoint), std::move(delimiter),
        std::move(discovered_tables));
  }
  catalog_set.SetDefaultGenerator(std::move(generator));

  (void)name;
  return std::move(catalog);
}

static unique_ptr<TransactionManager>
LanceStorageTransactionManager(optional_ptr<StorageExtensionInfo>,
                               AttachedDatabase &db, Catalog &) {
  return make_uniq<DuckTransactionManager>(db);
}

void RegisterLanceStorage(DBConfig &config) {
  auto ext = make_uniq<StorageExtension>();
  ext->attach = LanceStorageAttach;
  ext->create_transaction_manager = LanceStorageTransactionManager;
  config.storage_extensions["lance"] = std::move(ext);
}

} // namespace duckdb
