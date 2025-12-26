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
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"

#include "lance_common.hpp"

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

    auto view = make_uniq<CreateViewInfo>();
    view->schema = DEFAULT_SCHEMA;
    view->view_name = entry_name;
    view->sql = StringUtil::Format("SELECT * FROM lance_scan(%s)",
                                   SQLString(dataset_path));
    view->internal = true;
    view->on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    auto view_info = CreateViewInfo::FromSelect(context, std::move(view));
    return make_uniq_base<CatalogEntry, ViewCatalogEntry>(catalog, schema,
                                                          *view_info);
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
  auto catalog = make_uniq<DuckCatalog>(db);
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
    generator = make_uniq<LanceDirectoryDefaultGenerator>(
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
  auto &catalog_set = duck_schema.GetCatalogSet(CatalogType::VIEW_ENTRY);
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
