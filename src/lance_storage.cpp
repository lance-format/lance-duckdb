#include "duckdb/storage/storage_extension.hpp"

#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/view_catalog_entry.hpp"
#include "duckdb/catalog/default/default_generator.hpp"
#include "duckdb/catalog/default/default_schemas.hpp"
#include "duckdb/catalog/duck_catalog.hpp"
#include "duckdb/common/exception_format_value.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_table_entry.hpp"

#include "duckdb/function/table/arrow.hpp"

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

class LanceCatalog final : public DuckCatalog {
public:
  explicit LanceCatalog(AttachedDatabase &db) : DuckCatalog(db) {}

  PhysicalOperator &PlanInsert(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalInsert &op,
                               optional_ptr<PhysicalOperator> plan) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceInsertAppend(context, planner, op, plan);
    }
    return DuckCatalog::PlanInsert(context, planner, op, plan);
  }
};

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

class LanceDirectoryTableGenerator : public DefaultGenerator {
public:
  LanceDirectoryTableGenerator(Catalog &catalog, SchemaCatalogEntry &schema,
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

static unique_ptr<Catalog>
LanceStorageAttach(optional_ptr<StorageExtensionInfo>, ClientContext &context,
                   AttachedDatabase &db, const string &name, AttachInfo &info,
                   AttachOptions &) {
  auto attach_path = info.path;
  auto endpoint = GetLanceNamespaceEndpoint(info);
  auto delimiter = GetLanceNamespaceDelimiter(info);

  // Back the attached catalog by an in-memory DuckCatalog that lazily
  // materializes per-table views mapping to `lance_scan` / internal namespace
  // scan.
  info.path = ":memory:";
  auto catalog = make_uniq<LanceCatalog>(db);
  catalog->Initialize(false);

  unique_ptr<DefaultGenerator> generator;
  vector<string> discovered_tables;

  auto system_transaction =
      CatalogTransaction::GetSystemTransaction(db.GetDatabase());
  auto &schema = catalog->GetSchema(system_transaction, DEFAULT_SCHEMA);

  if (endpoint.empty()) {
    auto namespace_root =
        FileSystem::GetFileSystem(context).ExpandPath(attach_path);
    string list_error;
    if (!TryLanceDirNamespaceListTables(context, namespace_root,
                                        discovered_tables, list_error)) {
      throw IOException(
          "Failed to list tables from Lance directory namespace: " +
          list_error);
    }
    generator = make_uniq<LanceDirectoryTableGenerator>(
        *catalog, schema, std::move(namespace_root),
        std::move(discovered_tables));
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
    generator = make_uniq<LanceRestNamespaceDefaultGenerator>(
        *catalog, schema, std::move(endpoint), std::move(delimiter),
        std::move(discovered_tables));
  }

  auto &duck_schema = schema.Cast<DuckSchemaEntry>();
  auto &catalog_set = endpoint.empty()
                          ? duck_schema.GetCatalogSet(CatalogType::TABLE_ENTRY)
                          : duck_schema.GetCatalogSet(CatalogType::VIEW_ENTRY);
  catalog_set.SetDefaultGenerator(std::move(generator));

  (void)name;
  return std::move(catalog);
}

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
  if (txn.IsReadOnly()) {
    txn.SetReadWrite();
  }
  LancePendingAppend pending;
  pending.path = std::move(dataset_uri);
  pending.option_keys = std::move(option_keys);
  pending.option_values = std::move(option_values);
  pending.transaction = lance_transaction;
  tm->RegisterPendingAppend(txn, std::move(pending));
}

} // namespace duckdb
