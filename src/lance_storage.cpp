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
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_table_entry.hpp"

#include <cstring>

namespace duckdb {

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

static void PopulateLanceTableColumns(ClientContext &context,
                                      const string &dataset_path,
                                      ColumnList &out_columns) {
  auto *dataset = LanceOpenDataset(context, dataset_path);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + dataset_path +
                      LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    lance_close_dataset(dataset);
    throw IOException("Failed to get schema from Lance dataset: " +
                      dataset_path + LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    lance_close_dataset(dataset);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);
  lance_close_dataset(dataset);

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

class LanceDirectoryDefaultGenerator : public DefaultGenerator {
public:
  LanceDirectoryDefaultGenerator(Catalog &catalog, SchemaCatalogEntry &schema,
                                 string namespace_root,
                                 vector<string> discovered_tables)
      : DefaultGenerator(catalog), schema(schema),
        namespace_root(std::move(namespace_root)),
        discovered_tables(std::move(discovered_tables)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    auto &fs = FileSystem::GetFileSystem(context);
    auto dataset_dir = entry_name + ".lance";
    auto dataset_path = fs.JoinPath(namespace_root, dataset_dir);
    if (!fs.DirectoryExists(dataset_path)) {
      return nullptr;
    }
    // A Lance dataset directory is expected to have a `_versions` directory.
    // This avoids eagerly opening partially-created directories (e.g. leftovers
    // from previous failed runs).
    if (!fs.DirectoryExists(fs.JoinPath(dataset_path, "_versions"))) {
      return nullptr;
    }

    CreateTableInfo info(schema, entry_name);
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    PopulateLanceTableColumns(context, dataset_path, info.columns);
    return make_uniq_base<CatalogEntry, LanceTableEntry>(catalog, schema, info,
                                                         dataset_path);
  }

  vector<string> GetDefaultEntries() override { return discovered_tables; }

private:
  SchemaCatalogEntry &schema;
  string namespace_root;
  vector<string> discovered_tables;
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
    view->internal = true;
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
                   string directory_namespace_root, bool is_rest_namespace)
      : DuckSchemaEntry(catalog, info),
        directory_namespace_root(std::move(directory_namespace_root)),
        is_rest_namespace(is_rest_namespace) {}

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
};

class LanceDuckCatalog final : public DuckCatalog {
public:
  LanceDuckCatalog(AttachedDatabase &db, string directory_namespace_root,
                   bool is_rest_namespace)
      : DuckCatalog(db),
        directory_namespace_root(std::move(directory_namespace_root)),
        is_rest_namespace(is_rest_namespace) {}

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
    auto entry = make_uniq<LanceSchemaEntry>(
        *this, info, directory_namespace_root, is_rest_namespace);
    if (!schemas.CreateEntry(transaction, info.schema, std::move(entry),
                             dependencies)) {
      throw InternalException("Failed to replace Lance schema entry");
    }
  }

private:
  string directory_namespace_root;
  bool is_rest_namespace;
};

class PhysicalLanceTruncate final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceTruncate(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                        LanceTableEntry &table_p, idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p) {}

  bool IsSource() const override { return true; }

  class LanceTruncateSourceState final : public GlobalSourceState {
  public:
    bool emitted = false;
  };

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceTruncateSourceState>();
  }

  SourceResultType GetData(ExecutionContext &context, DataChunk &chunk,
                           OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceTruncateSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    auto row_count = LanceTruncateDataset(context.client, table.DatasetUri());
    chunk.SetCardinality(1);
    chunk.SetValue(0, 0, Value::BIGINT(row_count));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceTruncate"; }

private:
  LanceTableEntry &table;
};

static bool IsUnconditionalDelete(const LogicalDelete &op) {
  if (op.children.size() != 1) {
    return false;
  }
  return op.children[0]->type == LogicalOperatorType::LOGICAL_GET;
}

PhysicalOperator &LanceDuckCatalog::PlanDelete(ClientContext &context,
                                               PhysicalPlanGenerator &planner,
                                               LogicalDelete &op,
                                               PhysicalOperator &plan) {
  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    return DuckCatalog::PlanDelete(context, planner, op, plan);
  }
  if (op.return_chunk) {
    throw NotImplementedException(
        "Lance DELETE/TRUNCATE does not support RETURNING yet");
  }
  if (!IsUnconditionalDelete(op)) {
    throw NotImplementedException(
        "Lance DELETE only supports deleting all rows (use TRUNCATE TABLE)");
  }

  auto &truncate = planner.Make<PhysicalLanceTruncate>(
      op.types, *lance_table, op.estimated_cardinality);
  (void)context;
  (void)plan;
  return truncate;
}

static unique_ptr<Catalog>
LanceStorageAttach(optional_ptr<StorageExtensionInfo>, ClientContext &context,
                   AttachedDatabase &db, const string &name, AttachInfo &info,
                   AttachOptions &) {
  auto attach_path = info.path;
  auto endpoint = GetLanceNamespaceEndpoint(info);
  auto delimiter = GetLanceNamespaceDelimiter(info);

  unique_ptr<DefaultGenerator> generator;
  vector<string> discovered_tables;

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
  // materializes per-table entries mapping to `lance_scan` / internal namespace
  // scan, and supports CREATE TABLE for directory namespaces.
  info.path = ":memory:";
  auto catalog = make_uniq<LanceDuckCatalog>(db, directory_namespace_root,
                                             is_rest_namespace);
  catalog->Initialize(false);

  auto system_transaction =
      CatalogTransaction::GetSystemTransaction(db.GetDatabase());
  catalog->ReplaceDefaultSchemaWithLanceSchema(system_transaction);
  auto &schema = catalog->GetSchema(system_transaction, DEFAULT_SCHEMA);

  auto &duck_schema = schema.Cast<DuckSchemaEntry>();
  auto &catalog_set = duck_schema.GetCatalogSet(
      is_rest_namespace ? CatalogType::VIEW_ENTRY : CatalogType::TABLE_ENTRY);

  if (!is_rest_namespace) {
    generator = make_uniq<LanceDirectoryDefaultGenerator>(
        *catalog, schema, std::move(directory_namespace_root),
        std::move(discovered_tables));
  } else {
    generator = make_uniq<LanceRestNamespaceDefaultGenerator>(
        *catalog, schema, std::move(endpoint), std::move(delimiter),
        std::move(discovered_tables));
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
      key_ptrs.reserve(pending.option_keys.size());
      value_ptrs.reserve(pending.option_values.size());
      for (idx_t j = 0; j < pending.option_keys.size(); j++) {
        key_ptrs.push_back(pending.option_keys[j].c_str());
        value_ptrs.push_back(pending.option_values[j].c_str());
      }

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
