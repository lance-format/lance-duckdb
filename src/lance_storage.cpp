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
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
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

struct LanceRestNamespaceConfig {
  string endpoint;
  string namespace_id;
  string delimiter;
  string bearer_token_override;
  string api_key_override;
  string headers_tsv; // Tab-separated key\tvalue pairs for custom headers
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

// Parse HEADER options from ATTACH command
// Options like HEADER 'x-lancedb-database=lance_ns;x-api-key=sk_123' are parsed
// Multiple headers can be separated by semicolons within a single HEADER option
// Returns a TSV string with key\tvalue pairs separated by newlines
static string GetLanceNamespaceHeaders(const AttachInfo &info) {
  string headers_tsv;
  for (auto &kv : info.options) {
    // Handle 'HEADER' option with 'key=value' format
    if (StringUtil::CIEquals(kv.first, "header") && !kv.second.IsNull()) {
      auto header_str =
          kv.second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
      // Split by semicolon to support multiple headers
      vector<string> header_parts;
      size_t pos = 0;
      while (pos < header_str.size()) {
        auto next_semi = header_str.find(';', pos);
        if (next_semi == string::npos) {
          header_parts.push_back(header_str.substr(pos));
          break;
        }
        header_parts.push_back(header_str.substr(pos, next_semi - pos));
        pos = next_semi + 1;
      }
      for (auto &part : header_parts) {
        // Trim whitespace
        while (!part.empty() && isspace(part.front())) {
          part.erase(part.begin());
        }
        while (!part.empty() && isspace(part.back())) {
          part.pop_back();
        }
        auto eq_pos = part.find('=');
        if (eq_pos != string::npos && eq_pos > 0) {
          auto key = part.substr(0, eq_pos);
          auto value = part.substr(eq_pos + 1);
          if (!headers_tsv.empty()) {
            headers_tsv += "\n";
          }
          headers_tsv += key + "\t" + value;
        }
      }
    }
  }
  return headers_tsv;
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

  ArrowTableSchema arrow_table;
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table,
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

static vector<string>
ListRestNamespaceTables(const string &endpoint, const string &namespace_id,
                        const string &bearer_token, const string &api_key,
                        const string &delimiter, const string &headers_tsv) {
  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  auto *ptr = lance_namespace_list_tables(
      endpoint.c_str(), namespace_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr);
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
      string delimiter, string bearer_token_override, string api_key_override,
      string headers_tsv)
      : DefaultGenerator(catalog), schema(schema),
        endpoint(std::move(endpoint)), namespace_id(std::move(namespace_id)),
        bearer_token(std::move(bearer_token)), api_key(std::move(api_key)),
        delimiter(std::move(delimiter)),
        bearer_token_override(std::move(bearer_token_override)),
        api_key_override(std::move(api_key_override)),
        headers_tsv(std::move(headers_tsv)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    unordered_map<string, Value> overrides;
    if (!bearer_token_override.empty()) {
      overrides["bearer_token"] = Value(bearer_token_override);
    }
    if (!api_key_override.empty()) {
      overrides["api_key"] = Value(api_key_override);
    }

    string resolved_bearer;
    string resolved_api_key;
    ResolveLanceNamespaceAuth(context, endpoint, overrides, resolved_bearer,
                              resolved_api_key);
    // Backward-compatible fallback to credentials resolved during ATTACH.
    if (resolved_bearer.empty() && resolved_api_key.empty() &&
        (!bearer_token.empty() || !api_key.empty())) {
      resolved_bearer = bearer_token;
      resolved_api_key = api_key;
    }

    auto candidate_table_id = entry_name;
    string fallback_table_id;
    if (!namespace_id.empty()) {
      auto delim = delimiter.empty() ? "$" : delimiter;
      auto prefix = namespace_id + delim;
      if (!StringUtil::StartsWith(entry_name, prefix)) {
        fallback_table_id = prefix + entry_name;
      }
    }

    string table_id = candidate_table_id;
    string table_uri;
    auto *dataset = LanceOpenDatasetInNamespace(
        context, endpoint, candidate_table_id, resolved_bearer,
        resolved_api_key, delimiter, headers_tsv, table_uri);
    if (!dataset && !fallback_table_id.empty()) {
      table_id = fallback_table_id;
      dataset = LanceOpenDatasetInNamespace(context, endpoint, table_id,
                                            resolved_bearer, resolved_api_key,
                                            delimiter, headers_tsv, table_uri);
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

    LanceNamespaceTableConfig cfg;
    cfg.endpoint = endpoint;
    cfg.table_id = table_id;
    cfg.delimiter = delimiter;
    cfg.bearer_token_override = bearer_token_override;
    cfg.api_key_override = api_key_override;
    cfg.headers_tsv = headers_tsv;
    return make_uniq_base<CatalogEntry, LanceTableEntry>(catalog, schema, info,
                                                         std::move(cfg));
  }

  vector<string> GetDefaultEntries() override {
    auto tables = ListRestNamespaceTables(endpoint, namespace_id, bearer_token,
                                          api_key, delimiter, headers_tsv);
    if (namespace_id.empty()) {
      return tables;
    }
    auto delim = delimiter.empty() ? "$" : delimiter;
    auto prefix = namespace_id + delim;
    for (auto &t : tables) {
      if (StringUtil::StartsWith(t, prefix)) {
        t = t.substr(prefix.size());
      }
    }
    return tables;
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
  string headers_tsv;
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
                   shared_ptr<LanceRestNamespaceConfig> rest_ns)
      : DuckSchemaEntry(catalog, info), directory_ns(std::move(directory_ns)),
        rest_ns(std::move(rest_ns)) {}

  void SetTableDefaultGenerator(DefaultGenerator *generator) {
    table_default_generator = generator;
  }

  void Alter(CatalogTransaction transaction, AlterInfo &info) override {
    auto &set = GetCatalogSet(info.GetCatalogType());
    auto entry = set.GetEntry(transaction, info.name);
    auto *lance_entry =
        entry ? dynamic_cast<LanceTableEntry *>(entry.get()) : nullptr;

    if (!lance_entry) {
      DuckSchemaEntry::Alter(transaction, info);
      return;
    }

    auto &context = transaction.GetContext();
    if (!context.transaction.IsAutoCommit()) {
      throw NotImplementedException(
          "Lance DDL does not support explicit transactions yet");
    }

    // Allow altering internal entries for attached Lance catalogs.
    info.allow_internal = true;

    if (info.type == AlterType::SET_COMMENT) {
      auto &comment = info.Cast<SetCommentInfo>();
      const char *comment_ptr = nullptr;
      string comment_str;
      if (!comment.comment_value.IsNull()) {
        comment_str = comment.comment_value.DefaultCastAs(LogicalType::VARCHAR)
                          .GetValue<string>();
        comment_ptr = comment_str.c_str();
      }

      string display_uri;
      void *dataset =
          LanceOpenDatasetForTable(context, *lance_entry, display_uri);
      if (!dataset) {
        throw IOException("Failed to open Lance dataset: " + display_uri +
                          LanceFormatErrorSuffix());
      }
      auto rc =
          lance_dataset_update_table_metadata(dataset, "comment", comment_ptr);
      lance_close_dataset(dataset);
      if (rc != 0) {
        throw IOException("Failed to update table comment in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
    }

    auto system_tx =
        CatalogTransaction::GetSystemTransaction(catalog.GetDatabase());
    system_tx.context = &context;

    if (info.type == AlterType::CHANGE_OWNERSHIP) {
      if (!set.AlterOwnership(system_tx, info.Cast<ChangeOwnershipInfo>())) {
        throw CatalogException("Couldn't change ownership!");
      }
      return;
    }

    if (!set.AlterEntry(system_tx, info.name, info)) {
      throw CatalogException::MissingEntry(info.GetCatalogType(), info.name,
                                           string());
    }
  }

  void DropEntry(ClientContext &context, DropInfo &info) override {
    if (info.type != CatalogType::TABLE_ENTRY) {
      DuckSchemaEntry::DropEntry(context, info);
      return;
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

    if (rest_ns) {
      unordered_map<string, Value> overrides;
      if (!rest_ns->bearer_token_override.empty()) {
        overrides["bearer_token"] = Value(rest_ns->bearer_token_override);
      }
      if (!rest_ns->api_key_override.empty()) {
        overrides["api_key"] = Value(rest_ns->api_key_override);
      }

      string bearer_token;
      string api_key;
      ResolveLanceNamespaceAuth(context, rest_ns->endpoint, overrides,
                                bearer_token, api_key);

      auto leaf_id = info.name;
      string prefixed_id;
      if (!rest_ns->namespace_id.empty()) {
        auto delim = rest_ns->delimiter.empty() ? "$" : rest_ns->delimiter;
        auto prefix = rest_ns->namespace_id + delim;
        if (!StringUtil::StartsWith(leaf_id, prefix)) {
          prefixed_id = prefix + leaf_id;
        }
      }

      vector<string> discovered;
      string list_error;
      if (!TryLanceNamespaceListTables(
              context, rest_ns->endpoint, rest_ns->namespace_id, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, discovered,
              list_error)) {
        throw IOException("Failed to list tables from Lance namespace: " +
                          (list_error.empty() ? "unknown error" : list_error));
      }
      string table_id_for_ops = prefixed_id.empty() ? leaf_id : prefixed_id;
      for (auto &t : discovered) {
        if (!prefixed_id.empty() && StringUtil::CIEquals(t, prefixed_id)) {
          table_id_for_ops = prefixed_id;
          break;
        }
        if (StringUtil::CIEquals(t, leaf_id)) {
          table_id_for_ops = leaf_id;
          break;
        }
      }

      string drop_error;
      if (!TryLanceNamespaceDropTable(
              context, rest_ns->endpoint, table_id_for_ops, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, drop_error)) {
        throw IOException("Failed to drop Lance table via namespace: " +
                          (drop_error.empty() ? "unknown error" : drop_error));
      }
    } else {
      if (!IsSafeDatasetTableName(info.name)) {
        throw InvalidInputException(
            "Unsafe Lance dataset name for DROP TABLE: " + info.name);
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

    // DropEntry with a system (committed) CatalogTransaction leaves a committed
    // tombstone behind. This blocks subsequent lazy discovery of a recreated
    // dataset with the same name, because CatalogSet::GetEntryDetailed will
    // find the tombstone and never consult the default generator. Since ATTACH
    // TYPE LANCE catalogs are ephemeral, we can eagerly clean up the entry
    // chain (old entry + tombstone).
    set.CleanupEntry(*existing_entry);

    InvalidateTableDefaults();
  }

  optional_ptr<CatalogEntry> CreateTable(CatalogTransaction transaction,
                                         BoundCreateTableInfo &info) override {
    auto &create_info = info.Base();
    if (create_info.temporary) {
      throw NotImplementedException(
          "Lance ATTACH TYPE LANCE does not support TEMPORARY tables");
    }
    if (!info.constraints.empty() || !create_info.constraints.empty()) {
      throw NotImplementedException(
          "Lance CREATE TABLE does not support constraints");
    }
    auto &context = transaction.GetContext();
    string dataset_path;
    vector<string> option_keys;
    vector<string> option_values;

    if (rest_ns) {
      unordered_map<string, Value> overrides;
      if (!rest_ns->bearer_token_override.empty()) {
        overrides["bearer_token"] = Value(rest_ns->bearer_token_override);
      }
      if (!rest_ns->api_key_override.empty()) {
        overrides["api_key"] = Value(rest_ns->api_key_override);
      }

      string bearer_token;
      string api_key;
      ResolveLanceNamespaceAuth(context, rest_ns->endpoint, overrides,
                                bearer_token, api_key);

      auto leaf_id = create_info.table;
      string prefixed_id;
      if (!rest_ns->namespace_id.empty()) {
        auto delim = rest_ns->delimiter.empty() ? "$" : rest_ns->delimiter;
        auto prefix = rest_ns->namespace_id + delim;
        if (!StringUtil::StartsWith(leaf_id, prefix)) {
          prefixed_id = prefix + leaf_id;
        }
      }

      vector<string> discovered;
      string list_error;
      if (!TryLanceNamespaceListTables(
              context, rest_ns->endpoint, rest_ns->namespace_id, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, discovered,
              list_error)) {
        throw IOException("Failed to list tables from Lance namespace: " +
                          (list_error.empty() ? "unknown error" : list_error));
      }
      bool exists = false;
      string existing_id;
      for (auto &t : discovered) {
        if (!prefixed_id.empty() && StringUtil::CIEquals(t, prefixed_id)) {
          exists = true;
          existing_id = prefixed_id;
          break;
        }
        if (StringUtil::CIEquals(t, leaf_id)) {
          exists = true;
          existing_id = leaf_id;
          break;
        }
      }
      auto table_id_for_ops =
          exists ? existing_id : (prefixed_id.empty() ? leaf_id : prefixed_id);
      if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
          exists) {
        InvalidateTableDefaults();
        return nullptr;
      }
      if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
          exists) {
        throw IOException("Lance table already exists: " + existing_id);
      }
      if (create_info.on_conflict == OnCreateConflict::REPLACE_ON_CONFLICT &&
          exists) {
        string drop_error;
        if (!TryLanceNamespaceDropTable(context, rest_ns->endpoint,
                                        table_id_for_ops, bearer_token, api_key,
                                        rest_ns->delimiter,
                                        rest_ns->headers_tsv, drop_error)) {
          throw IOException(
              "Failed to drop Lance table via namespace: " +
              (drop_error.empty() ? "unknown error" : drop_error));
        }
      }

      string create_error;
      if (!TryLanceNamespaceCreateEmptyTable(
              context, rest_ns->endpoint, table_id_for_ops, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, dataset_path,
              option_keys, option_values, create_error)) {
        // Best-effort fallback for namespace implementations that do not use
        // a qualified object identifier for tables in ListTables.
        if (!prefixed_id.empty() && table_id_for_ops == prefixed_id) {
          option_keys.clear();
          option_values.clear();
          dataset_path.clear();
          create_error.clear();
          if (!TryLanceNamespaceCreateEmptyTable(
                  context, rest_ns->endpoint, leaf_id, bearer_token, api_key,
                  rest_ns->delimiter, rest_ns->headers_tsv, dataset_path,
                  option_keys, option_values, create_error)) {
            throw IOException(
                "Failed to create Lance table via namespace: " +
                (create_error.empty() ? "unknown error" : create_error));
          }
          table_id_for_ops = leaf_id;
        } else {
          throw IOException(
              "Failed to create Lance table via namespace: " +
              (create_error.empty() ? "unknown error" : create_error));
        }
      }
      if (dataset_path.empty()) {
        throw IOException(
            "Failed to create Lance table via namespace: empty location");
      }
      dataset_path = LanceNormalizeS3Scheme(dataset_path);
    } else {
      if (!IsSafeDatasetTableName(create_info.table)) {
        throw InvalidInputException(
            "Unsafe Lance dataset name for CREATE TABLE: " + create_info.table);
      }
      if (!directory_ns || directory_ns->root.empty()) {
        throw InternalException("Lance directory namespace root is empty");
      }

      dataset_path = JoinNamespacePath(directory_ns->root,
                                       GetDatasetDirName(create_info.table));

      auto exists =
          DirectoryNamespaceTableExists(*directory_ns, create_info.table);
      if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
          exists) {
        InvalidateTableDefaults();
        return nullptr;
      }
      if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
          exists) {
        throw IOException("Lance dataset already exists: " + dataset_path);
      }

      option_keys = directory_ns->option_keys;
      option_values = directory_ns->option_values;
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

    // Best-effort persistence of DuckDB column defaults in Lance field
    // metadata. Lance itself does not currently expose defaults through
    // DuckDB's catalog, but we can still use the metadata during UPDATE
    // execution to resolve SET col = DEFAULT.
    if (create_info.columns.LogicalColumnCount() > 0) {
      void *dataset = nullptr;
      if (option_keys.empty()) {
        dataset = lance_open_dataset(dataset_path.c_str());
      } else {
        dataset = lance_open_dataset_with_storage_options(
            dataset_path.c_str(), key_ptrs.data(), value_ptrs.data(),
            option_keys.size());
      }
      if (dataset) {
        for (auto &col : create_info.columns.Logical()) {
          if (!col.HasDefaultValue()) {
            continue;
          }
          auto default_expr = col.DefaultValue().ToString();
          (void)lance_dataset_update_field_metadata(dataset, col.Name().c_str(),
                                                    "duckdb_default_expr",
                                                    default_expr.c_str());
        }
        lance_close_dataset(dataset);
      }
    }

    InvalidateTableDefaults();
    return nullptr;
  }

private:
  void InvalidateTableDefaults() {
    if (!table_default_generator) {
      return;
    }
    table_default_generator->created_all_entries = false;
  }

  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;
  DefaultGenerator *table_default_generator = nullptr;
};

class LanceDuckCatalog final : public DuckCatalog {
public:
  using DuckCatalog::PlanDelete;

  LanceDuckCatalog(AttachedDatabase &db,
                   shared_ptr<LanceDirectoryNamespaceConfig> directory_ns,
                   shared_ptr<LanceRestNamespaceConfig> rest_ns)
      : DuckCatalog(db), directory_ns(std::move(directory_ns)),
        rest_ns(std::move(rest_ns)) {}

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
    if (rest_ns) {
      class PhysicalLanceCreateTableAs final : public PhysicalOperator {
      public:
        PhysicalLanceCreateTableAs(PhysicalPlan &physical_plan,
                                   vector<LogicalType> types_p, string endpoint,
                                   string namespace_id, string delimiter,
                                   string bearer_token_override,
                                   string api_key_override, string headers_tsv,
                                   string table_name, string writer_mode,
                                   vector<string> column_names_p,
                                   vector<LogicalType> column_types_p,
                                   idx_t estimated_cardinality)
            : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                               std::move(types_p), estimated_cardinality),
              endpoint(std::move(endpoint)),
              namespace_id(std::move(namespace_id)),
              delimiter(std::move(delimiter)),
              bearer_token_override(std::move(bearer_token_override)),
              api_key_override(std::move(api_key_override)),
              headers_tsv(std::move(headers_tsv)),
              table_name(std::move(table_name)),
              writer_mode(std::move(writer_mode)),
              column_names(std::move(column_names_p)),
              column_types(std::move(column_types_p)) {}

        bool IsSink() const override { return true; }
        bool IsSource() const override { return true; }
        bool ParallelSink() const override { return false; }
        bool SinkOrderDependent() const override { return false; }

        struct GlobalState final : public GlobalSinkState {
          mutex lock;

          string endpoint;
          string namespace_id;
          string delimiter;
          string bearer_token_override;
          string api_key_override;
          string headers_tsv;
          string table_name;
          string writer_mode;

          string table_id;
          string open_path;
          vector<string> option_keys;
          vector<string> option_values;

          vector<string> column_names;
          vector<LogicalType> column_types;

          idx_t insert_count = 0;
          void *writer = nullptr;
          ArrowSchemaWrapper schema_root;

          explicit GlobalState(string endpoint_p, string namespace_id_p,
                               string delimiter_p, string bearer_override_p,
                               string api_override_p, string headers_tsv_p,
                               string table_name_p, string writer_mode_p,
                               vector<string> col_names_p,
                               vector<LogicalType> col_types_p)
              : endpoint(std::move(endpoint_p)),
                namespace_id(std::move(namespace_id_p)),
                delimiter(std::move(delimiter_p)),
                bearer_token_override(std::move(bearer_override_p)),
                api_key_override(std::move(api_override_p)),
                headers_tsv(std::move(headers_tsv_p)),
                table_name(std::move(table_name_p)),
                writer_mode(std::move(writer_mode_p)),
                column_names(std::move(col_names_p)),
                column_types(std::move(col_types_p)) {}

          ~GlobalState() override {
            if (writer) {
              lance_close_writer(writer);
              writer = nullptr;
            }
          }
        };

        unique_ptr<GlobalSinkState>
        GetGlobalSinkState(ClientContext &context) const override {
          auto state = make_uniq<GlobalState>(
              endpoint, namespace_id, delimiter, bearer_token_override,
              api_key_override, headers_tsv, table_name, writer_mode,
              column_names, column_types);

          auto props = context.GetClientProperties();
          memset(&state->schema_root.arrow_schema, 0,
                 sizeof(state->schema_root.arrow_schema));
          ArrowConverter::ToArrowSchema(&state->schema_root.arrow_schema,
                                        state->column_types,
                                        state->column_names, props);

          unordered_map<string, Value> overrides;
          if (!state->bearer_token_override.empty()) {
            overrides["bearer_token"] = Value(state->bearer_token_override);
          }
          if (!state->api_key_override.empty()) {
            overrides["api_key"] = Value(state->api_key_override);
          }

          string bearer_token;
          string api_key;
          ResolveLanceNamespaceAuth(context, state->endpoint, overrides,
                                    bearer_token, api_key);

          auto delim = state->delimiter.empty() ? "$" : state->delimiter;
          auto prefix = state->namespace_id.empty()
                            ? string()
                            : (state->namespace_id + delim);
          auto leaf_id = state->table_name;
          string prefixed_id;
          if (!prefix.empty() && !StringUtil::StartsWith(leaf_id, prefix)) {
            prefixed_id = prefix + leaf_id;
          }

          vector<string> discovered;
          string list_error;
          if (!TryLanceNamespaceListTables(
                  context, state->endpoint, state->namespace_id, bearer_token,
                  api_key, state->delimiter, state->headers_tsv, discovered,
                  list_error)) {
            throw IOException(
                "Failed to list tables from Lance namespace: " +
                (list_error.empty() ? "unknown error" : list_error));
          }

          state->table_id = prefixed_id.empty() ? leaf_id : prefixed_id;
          for (auto &t : discovered) {
            if (!prefixed_id.empty() && StringUtil::CIEquals(t, prefixed_id)) {
              state->table_id = prefixed_id;
              break;
            }
            if (StringUtil::CIEquals(t, leaf_id)) {
              state->table_id = leaf_id;
              break;
            }
          }

          // If overwriting, drop any existing table first.
          if (state->writer_mode == "overwrite") {
            string drop_error;
            if (!TryLanceNamespaceDropTable(context, state->endpoint,
                                            state->table_id, bearer_token,
                                            api_key, state->delimiter,
                                            state->headers_tsv, drop_error)) {
              throw IOException(
                  "Failed to drop Lance table via namespace: " +
                  (drop_error.empty() ? "unknown error" : drop_error));
            }
          }

          string create_error;
          if (!TryLanceNamespaceCreateEmptyTable(
                  context, state->endpoint, state->table_id, bearer_token,
                  api_key, state->delimiter, state->headers_tsv,
                  state->open_path, state->option_keys, state->option_values,
                  create_error)) {
            if (!prefixed_id.empty() && state->table_id == prefixed_id) {
              state->table_id = leaf_id;
              state->open_path.clear();
              state->option_keys.clear();
              state->option_values.clear();
              create_error.clear();
              if (!TryLanceNamespaceCreateEmptyTable(
                      context, state->endpoint, state->table_id, bearer_token,
                      api_key, state->delimiter, state->headers_tsv,
                      state->open_path, state->option_keys,
                      state->option_values, create_error)) {
                throw IOException(
                    "Failed to create Lance table via namespace: " +
                    (create_error.empty() ? "unknown error" : create_error));
              }
            } else {
              throw IOException(
                  "Failed to create Lance table via namespace: " +
                  (create_error.empty() ? "unknown error" : create_error));
            }
          }
          if (state->open_path.empty()) {
            throw IOException(
                "Failed to create Lance table via namespace: empty location");
          }
          state->open_path = LanceNormalizeS3Scheme(state->open_path);

          vector<const char *> key_ptrs;
          vector<const char *> value_ptrs;
          BuildStorageOptionPointerArrays(
              state->option_keys, state->option_values, key_ptrs, value_ptrs);

          state->writer = lance_open_writer_with_storage_options(
              state->open_path.c_str(), state->writer_mode.c_str(),
              key_ptrs.empty() ? nullptr : key_ptrs.data(),
              value_ptrs.empty() ? nullptr : value_ptrs.data(),
              state->option_keys.size(), LANCE_DEFAULT_MAX_ROWS_PER_FILE,
              LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
              LANCE_DEFAULT_MAX_BYTES_PER_FILE,
              &state->schema_root.arrow_schema);
          if (!state->writer) {
            throw IOException("Failed to open Lance writer: " +
                              state->open_path + LanceFormatErrorSuffix());
          }

          return std::move(state);
        }

        struct LocalState final : public LocalSinkState {};
        unique_ptr<LocalSinkState>
        GetLocalSinkState(ExecutionContext &) const override {
          return make_uniq<LocalState>();
        }

        SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                            OperatorSinkInput &input) const override {
          if (chunk.size() == 0) {
            return SinkResultType::NEED_MORE_INPUT;
          }

          auto &gstate = input.global_state.Cast<GlobalState>();
          lock_guard<mutex> guard(gstate.lock);

          unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
              extension_type_cast;
          auto props = context.client.GetClientProperties();

          ArrowArray array;
          memset(&array, 0, sizeof(array));
          ArrowConverter::ToArrowArray(chunk, &array, props,
                                       extension_type_cast);

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

        SinkCombineResultType
        Combine(ExecutionContext &, OperatorSinkCombineInput &) const override {
          return SinkCombineResultType::FINISHED;
        }

        SinkFinalizeType
        Finalize(Pipeline &, Event &, ClientContext &context,
                 OperatorSinkFinalizeInput &input) const override {
          (void)context;
          auto &gstate = input.global_state.Cast<GlobalState>();

          {
            lock_guard<mutex> guard(gstate.lock);
            auto rc = lance_writer_finish(gstate.writer);
            lance_close_writer(gstate.writer);
            gstate.writer = nullptr;
            if (rc != 0) {
              throw IOException("Failed to finalize Lance CTAS write" +
                                LanceFormatErrorSuffix());
            }
          }

          return SinkFinalizeType::READY;
        }

        class SourceState : public GlobalSourceState {
        public:
          bool emitted = false;
        };

        unique_ptr<GlobalSourceState>
        GetGlobalSourceState(ClientContext &) const override {
          return make_uniq<SourceState>();
        }

        SourceResultType
        GetDataInternal(ExecutionContext &, DataChunk &chunk,
                        OperatorSourceInput &input) const override {
          auto &state = input.global_state.Cast<SourceState>();
          if (state.emitted) {
            return SourceResultType::FINISHED;
          }
          state.emitted = true;

          auto &gstate = sink_state->Cast<GlobalState>();
          chunk.SetCardinality(1);
          chunk.SetValue(
              0, 0, Value::BIGINT(NumericCast<int64_t>(gstate.insert_count)));
          return SourceResultType::FINISHED;
        }

        string GetName() const override { return "LanceCreateTableAs"; }

      private:
        string endpoint;
        string namespace_id;
        string delimiter;
        string bearer_token_override;
        string api_key_override;
        string headers_tsv;
        string table_name;
        string writer_mode;
        vector<string> column_names;
        vector<LogicalType> column_types;
      };

      // Use LIST TABLES to implement conflict behavior in a side-effect-free
      // way.
      unordered_map<string, Value> overrides;
      if (!rest_ns->bearer_token_override.empty()) {
        overrides["bearer_token"] = Value(rest_ns->bearer_token_override);
      }
      if (!rest_ns->api_key_override.empty()) {
        overrides["api_key"] = Value(rest_ns->api_key_override);
      }
      string bearer_token;
      string api_key;
      ResolveLanceNamespaceAuth(context, rest_ns->endpoint, overrides,
                                bearer_token, api_key);

      vector<string> discovered;
      string list_error;
      if (!TryLanceNamespaceListTables(
              context, rest_ns->endpoint, rest_ns->namespace_id, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, discovered,
              list_error)) {
        throw IOException("Failed to list tables from Lance namespace: " +
                          (list_error.empty() ? "unknown error" : list_error));
      }

      auto delim = rest_ns->delimiter.empty() ? "$" : rest_ns->delimiter;
      auto prefix = rest_ns->namespace_id.empty()
                        ? string()
                        : (rest_ns->namespace_id + delim);
      auto leaf_id = create_info.table;
      string prefixed_id;
      if (!prefix.empty() && !StringUtil::StartsWith(leaf_id, prefix)) {
        prefixed_id = prefix + leaf_id;
      }

      bool exists = false;
      string existing_id;
      for (auto &t : discovered) {
        if (!prefixed_id.empty() && StringUtil::CIEquals(t, prefixed_id)) {
          exists = true;
          existing_id = prefixed_id;
          break;
        }
        if (StringUtil::CIEquals(t, leaf_id)) {
          exists = true;
          existing_id = leaf_id;
          break;
        }
      }

      if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
          exists) {
        return planner.Make<PhysicalEmptyResult>(op.types,
                                                 op.estimated_cardinality);
      }
      if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
          exists) {
        throw IOException("Lance table already exists: " + existing_id);
      }

      auto names = create_info.columns.GetColumnNames();
      auto types = create_info.columns.GetColumnTypes();
      string mode = CreateTableModeFromConflict(create_info.on_conflict);
      auto &create_as = planner.Make<PhysicalLanceCreateTableAs>(
          op.types, rest_ns->endpoint, rest_ns->namespace_id,
          rest_ns->delimiter, rest_ns->bearer_token_override,
          rest_ns->api_key_override, rest_ns->headers_tsv, create_info.table,
          mode, std::move(names), std::move(types), op.estimated_cardinality);
      create_as.children.push_back(plan);
      return create_as;
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
    auto entry =
        make_uniq<LanceSchemaEntry>(*this, info, directory_ns, rest_ns);
    if (!schemas.CreateEntry(transaction, info.schema, std::move(entry),
                             dependencies)) {
      throw InternalException("Failed to replace Lance schema entry");
    }
  }

private:
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;
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
                   AttachOptions &attach_options) {
  // Consume Lance-specific options from attach_options.options so that
  // DuckDB doesn't complain about unrecognized options when creating storage.
  attach_options.options.erase("endpoint");
  attach_options.options.erase("delimiter");
  attach_options.options.erase("header");
  attach_options.options.erase("bearer_token");
  attach_options.options.erase("api_key");

  auto attach_path = info.path;
  auto endpoint = GetLanceNamespaceEndpoint(info);
  auto delimiter = GetLanceNamespaceDelimiter(info);
  auto headers_tsv = GetLanceNamespaceHeaders(info);

  unique_ptr<DefaultGenerator> generator;
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;

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
    if (!TryLanceNamespaceListTables(
            context, endpoint, namespace_id, bearer_token, api_key, delimiter,
            headers_tsv, discovered_tables, list_error)) {
      throw IOException("Failed to list tables from Lance namespace: " +
                        list_error);
    }

    rest_ns = make_shared_ptr<LanceRestNamespaceConfig>();
    rest_ns->endpoint = endpoint;
    rest_ns->namespace_id = namespace_id;
    rest_ns->delimiter = delimiter;
    rest_ns->bearer_token_override = bearer_token_override;
    rest_ns->api_key_override = api_key_override;
    rest_ns->headers_tsv = headers_tsv;
  }

  // Back the attached catalog by an in-memory DuckCatalog that lazily
  // materializes per-table entries mapping to internal scan / namespace scan,
  // scan, and supports CREATE TABLE for directory namespaces.
  info.path = ":memory:";
  auto catalog = make_uniq<LanceDuckCatalog>(db, directory_ns, rest_ns);
  catalog->Initialize(false);

  auto system_transaction =
      CatalogTransaction::GetSystemTransaction(db.GetDatabase());
  catalog->ReplaceDefaultSchemaWithLanceSchema(system_transaction);
  auto &schema = catalog->GetSchema(system_transaction, DEFAULT_SCHEMA);

  auto &lance_schema = schema.Cast<LanceSchemaEntry>();
  auto &duck_schema = schema.Cast<DuckSchemaEntry>();
  auto &catalog_set = duck_schema.GetCatalogSet(CatalogType::TABLE_ENTRY);

  if (!is_rest_namespace) {
    generator = make_uniq<LanceDirectoryDefaultGenerator>(*catalog, schema,
                                                          directory_ns);
  } else {
    generator = make_uniq<LanceRestNamespaceDefaultGenerator>(
        *catalog, schema, endpoint, namespace_id, std::move(bearer_token),
        std::move(api_key), delimiter, bearer_token_override, api_key_override,
        headers_tsv);
  }
  auto *generator_ptr = generator.get();
  catalog_set.SetDefaultGenerator(std::move(generator));
  lance_schema.SetTableDefaultGenerator(generator_ptr);

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
  StorageExtension::Register(config, "lance", std::move(ext));
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
