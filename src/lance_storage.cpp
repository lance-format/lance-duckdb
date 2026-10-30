#include "duckdb/storage/storage_extension.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/view_catalog_entry.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/catalog/default/default_schemas.hpp"
#include "duckdb/catalog/duck_catalog.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception/catalog_exception.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/common/exception_format_value.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_merge_into.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction.hpp"

#include "lance_arrow_compat.hpp"
#include "lance_blob.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_delete.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_merge.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"
#include "lance_update.hpp"
#include "lance_write.hpp"

#include <cstring>
#include <memory>

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
      idx_t pos = 0;
      auto header_str_size = NumericCast<idx_t>(header_str.size());
      while (pos < header_str_size) {
        auto next_semi =
            header_str.find(';', NumericCast<string::size_type>(pos));
        if (next_semi == string::npos) {
          header_parts.push_back(
              header_str.substr(NumericCast<string::size_type>(pos)));
          break;
        }
        auto next_semi_idx = NumericCast<idx_t>(next_semi);
        header_parts.push_back(header_str.substr(
            NumericCast<string::size_type>(pos),
            NumericCast<string::size_type>(next_semi_idx - pos)));
        pos = next_semi_idx + 1;
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

  LanceStringList tables = {};
  auto rc = lance_dir_namespace_list_tables(
      ns.root.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), ns.option_keys.size(),
      &tables);
  if (rc != 0) {
    throw IOException("Failed to list tables from Lance directory namespace: " +
                      ns.root + LanceFormatErrorSuffix());
  }
  return LanceConsumeStringList(tables);
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

  LanceStringList tables = {};
  auto rc = lance_namespace_list_tables(endpoint.c_str(), namespace_id.c_str(),
                                        bearer_ptr, api_key_ptr, delimiter_ptr,
                                        headers_ptr, &tables);
  if (rc != 0) {
    throw IOException("Failed to list tables from Lance namespace: " +
                      endpoint + "/" + namespace_id + LanceFormatErrorSuffix());
  }
  return LanceConsumeStringList(tables);
}

static string GetRestNamespaceTableId(const string &namespace_id,
                                      const string &table_name,
                                      const string &delimiter) {
  auto separator = delimiter.empty() ? "$" : delimiter;
  if (table_name.empty() || table_name.find(separator) != string::npos) {
    throw InvalidInputException(
        "Lance namespace table names must be non-empty and cannot contain "
        "the namespace delimiter '%s'",
        separator);
  }
  if (namespace_id.empty() || namespace_id == separator) {
    return table_name;
  }
  return namespace_id + separator + table_name;
}

static bool FindRestNamespaceTableId(const string &namespace_id,
                                     const string &table_name,
                                     const string &delimiter,
                                     const vector<string> &listed_names,
                                     string &out_table_id) {
  out_table_id = GetRestNamespaceTableId(namespace_id, table_name, delimiter);
  const string *matched = nullptr;
  for (const auto &name : listed_names) {
    if (!StringUtil::CIEquals(name, table_name)) {
      continue;
    }
    if (matched && *matched != name) {
      throw CatalogException(
          "Ambiguous Lance namespace table names '%s' and '%s'", *matched,
          name);
    }
    matched = &name;
  }
  if (!matched) {
    return false;
  }
  out_table_id = GetRestNamespaceTableId(namespace_id, *matched, delimiter);
  return true;
}

static string
FindDirectoryNamespaceTable(const LanceDirectoryNamespaceConfig &ns,
                            const string &table_name) {
  if (!IsSafeDatasetTableName(table_name)) {
    throw InvalidInputException(
        "Unsafe Lance dataset name for directory namespace: " + table_name);
  }
  auto tables = ListDirectoryNamespaceTables(ns);
  const Identifier requested(table_name);
  string matched;
  for (auto &t : tables) {
    if (Identifier(t) == requested) {
      if (!matched.empty()) {
        throw CatalogException(
            "Ambiguous Lance directory table names '%s' and '%s' in '%s'",
            matched, t, ns.root);
      }
      matched = t;
    }
  }
  return matched;
}

// Namespace enumeration needs the current context for credential resolution.
// DuckDB's DefaultGenerator only supports context-free enumeration.
class LanceTableGenerator {
public:
  explicit LanceTableGenerator(Catalog &catalog) : catalog(catalog) {}
  virtual ~LanceTableGenerator() = default;

  virtual unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context, const Identifier &entry_name) = 0;
  virtual vector<Identifier> GetDefaultEntries(ClientContext &context) = 0;

protected:
  Catalog &catalog;
};

class LanceDirectoryDefaultGenerator : public LanceTableGenerator {
public:
  LanceDirectoryDefaultGenerator(Catalog &catalog, SchemaCatalogEntry &schema,
                                 shared_ptr<LanceDirectoryNamespaceConfig> ns)
      : LanceTableGenerator(catalog), schema(schema), ns(std::move(ns)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const Identifier &entry_name_p) override {
    const auto &entry_name = entry_name_p.GetIdentifierName();
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

    CreateTableInfo info(schema, Identifier(entry_name));
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    vector<string> coerced;
    vector<string> json_columns;
    vector<bool> blob_columns;
    vector<string> writable_blob_v2_columns;
    try {
      auto live = LanceTableSchema::FromDataset(context, dataset);
      info.columns = std::move(live.columns);
      info.constraints = std::move(live.constraints);
      coerced = std::move(live.coerced_columns);
      json_columns = std::move(live.json_columns);
      blob_columns = std::move(live.blob_columns);
      writable_blob_v2_columns = std::move(live.writable_blob_v2_columns);
    } catch (...) {
      lance_close_dataset(dataset);
      return nullptr;
    }
    lance_close_dataset(dataset);

    LanceNamespaceTableConfig cfg;
    cfg.kind = LanceNamespaceKind::Directory;
    cfg.root = ns->root;
    cfg.table_id = entry_name;
    cfg.option_keys = ns->option_keys;
    cfg.option_values = ns->option_values;
    cfg.display_uri = dataset_uri.empty()
                          ? LanceDirectoryNamespaceDatasetUri(cfg)
                          : std::move(dataset_uri);
    auto entry =
        make_uniq<LanceTableEntry>(catalog, schema, info, std::move(cfg));
    entry->SetCoercedColumnNames(std::move(coerced));
    entry->SetJsonColumnNames(std::move(json_columns));
    entry->SetBlobColumns(std::move(blob_columns));
    entry->SetWritableBlobV2Columns(std::move(writable_blob_v2_columns));
    return unique_ptr_cast<LanceTableEntry, CatalogEntry>(std::move(entry));
  }

  vector<Identifier> GetDefaultEntries(ClientContext &) override {
    if (!ns) {
      return {};
    }
    auto all = ListDirectoryNamespaceTables(*ns);
    identifier_set_t physical_names;
    for (auto &name : all) {
      auto inserted = physical_names.emplace(name);
      if (!inserted.second) {
        throw CatalogException(
            "Ambiguous Lance directory table names '%s' and '%s' in '%s'",
            inserted.first->GetIdentifierName(), name, ns->root);
      }
    }
    // Catalog scans require every listed table to produce an entry. Filter
    // unreadable datasets here, while point lookup can still return nullptr
    // for a dataset that CTAS has not written yet.
    vector<Identifier> valid;
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(ns->option_keys, ns->option_values,
                                    key_ptrs, value_ptrs);
    for (auto &name : all) {
      const char *uri_ptr = nullptr;
      auto *ds = lance_open_dataset_in_dir_namespace(
          ns->root.c_str(), name.c_str(),
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(),
          ns->option_keys.size(), &uri_ptr);
      if (uri_ptr) {
        lance_free_string(uri_ptr);
      }
      if (ds) {
        lance_close_dataset(ds);
        valid.emplace_back(std::move(name));
      }
    }
    return valid;
  }

private:
  SchemaCatalogEntry &schema;
  shared_ptr<LanceDirectoryNamespaceConfig> ns;
};

static LanceTableSchema LanceTableSchemaFromJson(ClientContext &context,
                                                 const string &schema_json) {
  ArrowSchemaWrapper root;
  root.arrow_schema = {};
  if (lance_json_arrow_schema_to_c(schema_json.c_str(), &root.arrow_schema) !=
      0) {
    throw IOException(
        "Failed to convert JSON Arrow schema to C Data Interface" +
        LanceFormatErrorSuffix());
  }
  ArrowTableSchema arrow_table;
  return LanceTableSchema::FromArrow(context, root.arrow_schema, arrow_table);
}

class LanceRestNamespaceDefaultGenerator : public LanceTableGenerator {
public:
  LanceRestNamespaceDefaultGenerator(
      Catalog &catalog, SchemaCatalogEntry &schema, string endpoint,
      string namespace_id, string delimiter, string bearer_token_override,
      string api_key_override, string headers_tsv)
      : LanceTableGenerator(catalog), schema(schema),
        endpoint(std::move(endpoint)), namespace_id(std::move(namespace_id)),
        delimiter(std::move(delimiter)),
        bearer_token_override(std::move(bearer_token_override)),
        api_key_override(std::move(api_key_override)),
        headers_tsv(std::move(headers_tsv)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const Identifier &entry_name_p) override {
    const auto &entry_name = entry_name_p.GetIdentifierName();
    auto table_id =
        GetRestNamespaceTableId(namespace_id, entry_name, delimiter);
    string bearer_token;
    string api_key;
    ResolveAuth(context, bearer_token, api_key);

    const char *schema_ptr = nullptr;
    auto rc = lance_namespace_describe_table_with_schema(
        endpoint.c_str(), table_id.c_str(),
        bearer_token.empty() ? nullptr : bearer_token.c_str(),
        api_key.empty() ? nullptr : api_key.c_str(),
        delimiter.empty() ? nullptr : delimiter.c_str(),
        headers_tsv.empty() ? nullptr : headers_tsv.c_str(), &schema_ptr);
    std::unique_ptr<const char, decltype(&lance_free_string)> schema_owner(
        schema_ptr, lance_free_string);
    if (rc != 0) {
      auto code = lance_last_error_code();
      auto error = LanceFormatErrorSuffix();
      if (code == LANCE_FFI_NAMESPACE_TABLE_NOT_FOUND) {
        return nullptr;
      }
      throw IOException(
          "Failed to describe Lance namespace table: " + table_id + error);
    }
    if (schema_ptr) {
      return MakeNamespaceEntry(entry_name, table_id,
                                LanceTableSchemaFromJson(context, schema_ptr));
    }

    string table_uri;
    std::unique_ptr<void, decltype(&lance_close_dataset)> dataset(
        LanceOpenDatasetInNamespace(context, endpoint, table_id, bearer_token,
                                    api_key, delimiter, headers_tsv, table_uri),
        lance_close_dataset);
    if (!dataset) {
      auto code = lance_last_error_code();
      auto error = LanceFormatErrorSuffix();
      if (code == LANCE_FFI_NAMESPACE_TABLE_NOT_FOUND) {
        return nullptr;
      }
      throw IOException("Failed to open Lance namespace table: " + table_id +
                        error);
    }
    return MakeNamespaceEntry(
        entry_name, table_id,
        LanceTableSchema::FromDataset(context, dataset.get()));
  }

  vector<Identifier> GetDefaultEntries(ClientContext &context) override {
    string bearer_token;
    string api_key;
    ResolveAuth(context, bearer_token, api_key);
    auto names = ListRestNamespaceTables(endpoint, namespace_id, bearer_token,
                                         api_key, delimiter, headers_tsv);
    return vector<Identifier>(names.begin(), names.end());
  }

private:
  void ResolveAuth(ClientContext &context, string &bearer_token,
                   string &api_key) const {
    unordered_map<string, Value> overrides;
    if (!bearer_token_override.empty()) {
      overrides["bearer_token"] = Value(bearer_token_override);
    }
    if (!api_key_override.empty()) {
      overrides["api_key"] = Value(api_key_override);
    }
    ResolveLanceNamespaceAuth(context, endpoint, overrides, bearer_token,
                              api_key);
  }

  unique_ptr<CatalogEntry> MakeNamespaceEntry(const string &entry_name,
                                              const string &table_id,
                                              LanceTableSchema live) {
    CreateTableInfo info(schema, Identifier(entry_name));
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    info.columns = std::move(live.columns);
    info.constraints = std::move(live.constraints);
    LanceNamespaceTableConfig cfg;
    cfg.kind = LanceNamespaceKind::Rest;
    cfg.endpoint = endpoint;
    cfg.table_id = table_id;
    cfg.delimiter = delimiter;
    cfg.bearer_token_override = bearer_token_override;
    cfg.api_key_override = api_key_override;
    cfg.headers_tsv = headers_tsv;
    auto entry =
        make_uniq<LanceTableEntry>(catalog, schema, info, std::move(cfg));
    entry->SetCoercedColumnNames(std::move(live.coerced_columns));
    entry->SetJsonColumnNames(std::move(live.json_columns));
    entry->SetBlobColumns(std::move(live.blob_columns));
    entry->SetWritableBlobV2Columns(std::move(live.writable_blob_v2_columns));
    return unique_ptr_cast<LanceTableEntry, CatalogEntry>(std::move(entry));
  }

  // Mirrored by ErrorCode::NamespaceTableNotFound in rust/error.rs.
  static constexpr int32_t LANCE_FFI_NAMESPACE_TABLE_NOT_FOUND = 57;
  SchemaCatalogEntry &schema;
  string endpoint;
  string namespace_id;
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

static string
GetCreateTableDataStorageVersionOption(const CreateTableInfo &create_info) {
  auto it = create_info.options.find("data_storage_version");
  if (it == create_info.options.end()) {
    return LANCE_DEFAULT_DATA_STORAGE_VERSION;
  }
  if (!it->second) {
    throw BinderException("data_storage_version option cannot be NULL");
  }
  auto &expr = *it->second;
  if (expr.GetExpressionClass() != ExpressionClass::CONSTANT) {
    throw BinderException(
        "data_storage_version option must be a constant string");
  }
  auto &constant = expr.Cast<ConstantExpression>();
  if (constant.GetLiteral().ToValue().IsNull()) {
    throw BinderException("data_storage_version option cannot be NULL");
  }
  auto value = constant.GetLiteral()
                   .ToValue()
                   .DefaultCastAs(LogicalType::VARCHAR)
                   .GetValue<string>();
  StringUtil::Trim(value);
  if (value.empty()) {
    throw BinderException("data_storage_version option cannot be empty");
  }
  return value;
}

// The namespace is authoritative; these entries are an in-memory metadata
// cache, not DuckDB native storage. Queries pin every generation they bind so a
// refresh, ALTER or DROP cannot destroy an entry still used by another
// executing plan.
class LanceCatalogQueryState final : public ClientContextState {
public:
  void Pin(const shared_ptr<CatalogEntry> &entry) {
    entries.emplace(entry.get(), entry);
  }
  void QueryEnd() override { entries.clear(); }

private:
  unordered_map<CatalogEntry *, shared_ptr<CatalogEntry>> entries;
};

static optional_ptr<CatalogEntry>
PinLanceCatalogEntry(ClientContext &context,
                     const shared_ptr<CatalogEntry> &entry) {
  context.registered_state
      ->GetOrCreate<LanceCatalogQueryState>("lance_catalog_entries")
      ->Pin(entry);
  return entry.get();
}

class LanceSchemaEntry final : public DuckSchemaEntry {
public:
  LanceSchemaEntry(Catalog &catalog, CreateSchemaInfo &info,
                   shared_ptr<LanceDirectoryNamespaceConfig> directory_ns,
                   shared_ptr<LanceRestNamespaceConfig> rest_ns)
      : DuckSchemaEntry(catalog, info), directory_ns(std::move(directory_ns)),
        rest_ns(std::move(rest_ns)) {}

  void SetTableDefaultGenerator(unique_ptr<LanceTableGenerator> generator) {
    table_default_generator = std::move(generator);
  }

  void Alter(CatalogTransaction transaction, AlterInfo &info) override {
    auto entry = GetEntry(transaction, info.GetCatalogType(),
                          info.GetQualifiedName().Name());
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

    lock_guard<mutex> guard(table_lock);
    auto current = tables.find(lance_entry->name);
    if (current == tables.end() || current->second.get() != lance_entry) {
      throw TransactionException(
          "Lance table changed concurrently; retry ALTER");
    }

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
      LanceInvalidateDatasetCacheForTable(context, *lance_entry);
    }

    auto updated = info.type == AlterType::SET_COMMENT
                       ? lance_entry->Rebuild(context)
                       : lance_entry->AlterEntry(context, info);
    if (updated) {
      current->second = shared_ptr<CatalogEntry>(std::move(updated));
    }
  }

  optional_ptr<CatalogEntry>
  LookupEntry(CatalogTransaction transaction,
              const EntryLookupInfo &lookup_info) override {
    return LookupEntryInternal(transaction, lookup_info);
  }

  CatalogSet::EntryLookup
  LookupEntryDetailed(CatalogTransaction transaction,
                      const EntryLookupInfo &info) override {
    return SchemaCatalogEntry::LookupEntryDetailed(transaction, info);
  }

  SimilarCatalogEntry GetSimilarEntry(CatalogTransaction transaction,
                                      const EntryLookupInfo &info) override {
    return SchemaCatalogEntry::GetSimilarEntry(transaction, info);
  }

  // Keep the native context-free scan used by DuckDB checkpoints.
  using DuckSchemaEntry::Scan;

  void Scan(CatalogTransaction transaction, CatalogType type,
            const std::function<void(CatalogEntry &)> &callback) override {
    DuckSchemaEntry::Scan(transaction, type, callback);
    if (type != CatalogType::TABLE_ENTRY || !transaction.context) {
      return;
    }
    vector<Identifier> names;
    {
      lock_guard<mutex> guard(table_lock);
      names = table_default_generator->GetDefaultEntries(*transaction.context);
    }
    for (auto &name : names) {
      auto entry = LookupEntryInternal(
          transaction, EntryLookupInfo(type, QualifiedName(name)),
          directory_ns ? &name : nullptr);
      if (entry && dynamic_cast<LanceTableEntry *>(entry.get())) {
        callback(*entry);
      }
    }
  }

  optional_ptr<CatalogEntry> CreateView(CatalogTransaction transaction,
                                        CreateViewInfo &info) override {
    auto entry =
        GetEntry(transaction, CatalogType::TABLE_ENTRY, info.GetViewName());
    if (entry && dynamic_cast<LanceTableEntry *>(entry.get())) {
      if (info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT) {
        return nullptr;
      }
      throw CatalogException("Cannot replace a Lance table with a view");
    }
    return DuckSchemaEntry::CreateView(transaction, info);
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
    auto existing_entry =
        GetEntry(transaction, info.type, info.GetQualifiedName().Name());
    if (!existing_entry) {
      throw InternalException(
          "Failed to drop entry \"%s\" - entry could not be found",
          info.GetQualifiedName().Name());
    }
    auto *lance_entry = dynamic_cast<LanceTableEntry *>(existing_entry.get());
    if (!lance_entry) {
      DuckSchemaEntry::DropEntry(context, info);
      return;
    }
    auto cache_key = LanceBuildDatasetCacheKeyForTable(context, *lance_entry);
    lock_guard<mutex> guard(table_lock);
    auto current = tables.find(lance_entry->name);
    if (current == tables.end() || current->second.get() != lance_entry) {
      throw TransactionException(
          "Lance table changed concurrently; retry DROP");
    }

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

      const auto &table_id_for_ops = lance_entry->NamespaceConfig().table_id;

      string drop_error;
      if (!TryLanceNamespaceDropTable(
              context, rest_ns->endpoint, table_id_for_ops, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, drop_error)) {
        throw IOException("Failed to drop Lance table via namespace: " +
                          (drop_error.empty() ? "unknown error" : drop_error));
      }
    } else {
      if (!IsSafeDatasetTableName(
              info.GetQualifiedName().Name().GetIdentifierName())) {
        throw InvalidInputException(
            "Unsafe Lance dataset name for DROP TABLE: " +
            info.GetQualifiedName().Name());
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
          root.c_str(), lance_entry->NamespaceConfig().table_id.c_str(),
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size());
      if (rc != 0) {
        throw IOException(
            "Failed to drop Lance dataset: " + root + "/" +
            GetDatasetDirName(lance_entry->NamespaceConfig().table_id) +
            LanceFormatErrorSuffix());
      }
    }

    tables.erase(current);
    LanceInvalidateDatasetCache(context, cache_key);
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
    auto data_storage_version =
        GetCreateTableDataStorageVersionOption(create_info);
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

      vector<string> discovered;
      string list_error;
      if (!TryLanceNamespaceListTables(
              context, rest_ns->endpoint, rest_ns->namespace_id, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, discovered,
              list_error)) {
        throw IOException("Failed to list tables from Lance namespace: " +
                          (list_error.empty() ? "unknown error" : list_error));
      }
      string table_id_for_ops;
      auto exists = FindRestNamespaceTableId(
          rest_ns->namespace_id, create_info.GetTableName().GetIdentifierName(),
          rest_ns->delimiter, discovered, table_id_for_ops);
      if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
          exists) {
        InvalidateTableEntry(create_info.GetTableName());
        return nullptr;
      }
      if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
          exists) {
        throw IOException("Lance table already exists: " + table_id_for_ops);
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
        throw IOException(
            "Failed to create Lance table via namespace: " +
            (create_error.empty() ? "unknown error" : create_error));
      }
      if (dataset_path.empty()) {
        throw IOException(
            "Failed to create Lance table via namespace: empty location");
      }
      dataset_path = LanceNormalizeS3Scheme(dataset_path);
    } else {
      if (!IsSafeDatasetTableName(
              create_info.GetTableName().GetIdentifierName())) {
        throw InvalidInputException(
            "Unsafe Lance dataset name for CREATE TABLE: " +
            create_info.GetTableName().GetIdentifierName());
      }
      if (!directory_ns || directory_ns->root.empty()) {
        throw InternalException("Lance directory namespace root is empty");
      }

      auto physical_table = FindDirectoryNamespaceTable(
          *directory_ns, create_info.GetTableName().GetIdentifierName());
      auto exists = !physical_table.empty();
      if (!exists) {
        physical_table = create_info.GetTableName().GetIdentifierName();
      }
      dataset_path = JoinNamespacePath(directory_ns->root,
                                       GetDatasetDirName(physical_table));
      if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
          exists) {
        InvalidateTableEntry(create_info.GetTableName());
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
      names.push_back(col.Name().GetIdentifierName());
      types.push_back(col.Type());
    }

    ArrowSchemaWrapper schema_root;
    memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
    auto props = context.GetClientProperties();
    ArrowConverter::ToArrowSchema(&schema_root.arrow_schema, types, names,
                                  props);
    LanceNormalizeArrowListFieldNames(&schema_root.arrow_schema);

    auto mode = CreateTableModeFromConflict(create_info.on_conflict);
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                    value_ptrs);
    const char *data_storage_version_ptr =
        data_storage_version.empty() ? nullptr : data_storage_version.c_str();
    auto blob_flags = LanceBlobFlagsForTypes(types);

    auto *writer = lance_open_writer_with_storage_options(
        dataset_path.c_str(), mode.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
        LANCE_DEFAULT_MAX_ROWS_PER_FILE, LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
        LANCE_DEFAULT_MAX_BYTES_PER_FILE, data_storage_version_ptr,
        LanceGetSessionHandle(context), &schema_root.arrow_schema,
        blob_flags.empty() ? nullptr : blob_flags.data(), blob_flags.size(),
        LanceBlobIngestBatchBudget(context));
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
    // metadata. Catalog discovery restores these expressions so DuckDB can
    // bind INSERT and UPDATE DEFAULT through its native binder.
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

    InvalidateTableEntry(create_info.GetTableName());
    return nullptr;
  }

  bool ReplaceStaleTableEntry(CatalogTransaction transaction,
                              LanceTableEntry &table) {
    if (!transaction.context) {
      return false;
    }
    lock_guard<mutex> guard(table_lock);
    auto found = tables.find(table.name);
    if (found == tables.end() || found->second.get() != &table) {
      return false;
    }
    try {
      found->second =
          shared_ptr<CatalogEntry>(table.Rebuild(*transaction.context));
    } catch (IOException &) {
      return false;
    }
    return true;
  }

private:
  optional_ptr<CatalogEntry>
  LookupEntryInternal(CatalogTransaction transaction,
                      const EntryLookupInfo &lookup_info,
                      optional_ptr<const Identifier> physical_name = nullptr) {
    auto native = DuckSchemaEntry::LookupEntry(transaction, lookup_info);
    if (native || lookup_info.GetCatalogType() != CatalogType::TABLE_ENTRY ||
        !transaction.context) {
      return native;
    }
    lock_guard<mutex> guard(table_lock);
    auto &name = lookup_info.GetEntryIdentifier();
    auto found = tables.find(name);
    if (found == tables.end()) {
      Identifier backend_name = name;
      if (directory_ns) {
        if (physical_name) {
          backend_name = *physical_name;
        } else {
          auto matched = FindDirectoryNamespaceTable(*directory_ns,
                                                     name.GetIdentifierName());
          if (matched.empty()) {
            return nullptr;
          }
          backend_name = Identifier(std::move(matched));
        }
      }
      auto entry = table_default_generator->CreateDefaultEntry(
          *transaction.context, backend_name);
      if (!entry) {
        return nullptr;
      }
      found = tables.emplace(name, shared_ptr<CatalogEntry>(std::move(entry)))
                  .first;
    }
    {
      auto &table = found->second->Cast<LanceTableEntry>();
      try {
        if (table.IsSchemaStale(*transaction.context)) {
          found->second =
              shared_ptr<CatalogEntry>(table.Rebuild(*transaction.context));
        }
      } catch (IOException &) {
        // Preserve metadata access and DROP when storage is unavailable.
        // Scan and DML binding still validate the live schema and fail closed.
      }
    }
    return PinLanceCatalogEntry(*transaction.context, found->second);
  }

  void InvalidateTableEntry(const Identifier &name) {
    lock_guard<mutex> guard(table_lock);
    tables.erase(name);
  }

  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;
  unique_ptr<LanceTableGenerator> table_default_generator;
  mutex table_lock;
  identifier_map_t<shared_ptr<CatalogEntry>> tables;
};

bool LanceTryReplaceStaleTableEntry(ClientContext &context,
                                    LanceTableEntry &table) {
  auto *lance_schema = dynamic_cast<LanceSchemaEntry *>(&table.schema);
  if (!lance_schema) {
    return false;
  }
  auto transaction = table.ParentCatalog().GetCatalogTransaction(context);
  return lance_schema->ReplaceStaleTableEntry(transaction, table);
}

class LanceDuckCatalog final : public DuckCatalog {
public:
  using DuckCatalog::PlanDelete;
  using DuckCatalog::PlanMergeInto;

  LanceDuckCatalog(AttachedDatabase &db,
                   shared_ptr<LanceDirectoryNamespaceConfig> directory_ns,
                   shared_ptr<LanceRestNamespaceConfig> rest_ns)
      : DuckCatalog(db), directory_ns(std::move(directory_ns)),
        rest_ns(std::move(rest_ns)) {}

  using DuckCatalog::PlanUpdate;

  // Lance table entries mirror external datasets whose commits never bump any
  // DuckDB catalog version, so a "current" version number would vouch for
  // state this catalog cannot actually track. Returning an invalid version
  // opts the catalog out of catalog-version identity checks - the documented
  // escape hatch in CheckCatalogIdentity (prepared_statement_data.cpp) for
  // catalogs that don't support catalog versions - which forces every EXECUTE
  // of a prepared statement that reads or writes this catalog to rebind and
  // observe the same latest state as ad-hoc statements. Scans already force
  // this through LanceScanBindData::SupportStatementCache() returning false,
  // but target-only DML (INSERT/UPDATE/DELETE/MERGE without a scan of the
  // target, e.g. INSERT ... VALUES) binds no scan: its cached physical plan
  // pins a LanceTableEntry reference that would outlive a catalog-entry
  // replacement and bypass all bind-time freshness checks.
  optional_idx GetCatalogVersion(ClientContext &) override {
    return optional_idx();
  }

  ErrorData SupportsCreateTable(BoundCreateTableInfo &info) override {
    auto &base = info.Base().Cast<CreateTableInfo>();
    if (!base.partition_keys.empty()) {
      return ErrorData(ExceptionType::CATALOG,
                       StringUtil::Format("PARTITIONED BY is not supported "
                                          "for tables in a %s catalog",
                                          GetCatalogType()));
    }
    if (!base.sort_keys.empty()) {
      return ErrorData(
          ExceptionType::CATALOG,
          StringUtil::Format("SORTED BY is not supported for tables in a %s "
                             "catalog",
                             GetCatalogType()));
    }
    for (auto &entry : base.options) {
      if (!StringUtil::CIEquals(entry.first, "data_storage_version")) {
        return ErrorData(ExceptionType::CATALOG,
                         "Only data_storage_version is supported in WITH "
                         "clause for Lance tables");
      }
    }
    return ErrorData();
  }

  PhysicalOperator &PlanUpdate(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalUpdate &op,
                               PhysicalOperator &plan) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceUpdateOverwrite(context, planner, op, plan);
    }
    return DuckCatalog::PlanUpdate(context, planner, op, plan);
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

  PhysicalOperator &PlanMergeInto(ClientContext &context,
                                  PhysicalPlanGenerator &planner,
                                  LogicalMergeInto &op,
                                  PhysicalOperator &plan) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceMergeInto(context, planner, op, plan);
    }
    return DuckCatalog::PlanMergeInto(context, planner, op, plan);
  }

  PhysicalOperator &PlanCreateTableAs(ClientContext &context,
                                      PhysicalPlanGenerator &planner,
                                      LogicalCreateTable &op,
                                      PhysicalOperator &plan) override {
    auto &create_info = op.info->Base();
    auto data_storage_version =
        GetCreateTableDataStorageVersionOption(create_info);
    if (create_info.temporary) {
      throw NotImplementedException(
          "Lance ATTACH TYPE LANCE does not support TEMPORARY tables");
    }
    if (rest_ns) {
      class PhysicalLanceCreateTableAs final : public PhysicalOperator {
      public:
        PhysicalLanceCreateTableAs(
            PhysicalPlan &physical_plan, vector<LogicalType> types_p,
            string endpoint, string namespace_id, string delimiter,
            string bearer_token_override, string api_key_override,
            string headers_tsv, string table_name, string writer_mode,
            string data_storage_version, vector<string> column_names_p,
            vector<LogicalType> column_types_p, idx_t estimated_cardinality)
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
              data_storage_version(std::move(data_storage_version)),
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
          string data_storage_version;

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
                               string data_storage_version_p,
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
                data_storage_version(std::move(data_storage_version_p)),
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
              data_storage_version, column_names, column_types);

          auto props = context.GetClientProperties();
          memset(&state->schema_root.arrow_schema, 0,
                 sizeof(state->schema_root.arrow_schema));
          ArrowConverter::ToArrowSchema(&state->schema_root.arrow_schema,
                                        state->column_types,
                                        state->column_names, props);
          LanceNormalizeArrowListFieldNames(&state->schema_root.arrow_schema);

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
          FindRestNamespaceTableId(state->namespace_id, state->table_name,
                                   state->delimiter, discovered,
                                   state->table_id);

          // REST namespace metadata changes (DROP + CREATE empty table) happen
          // before the writer starts and are not part of the Lance data
          // transaction. A failed write therefore publishes no Lance data
          // version, but an overwrite cannot roll the namespace catalog entry
          // back; this is the existing REST service boundary, not a
          // distributed namespace transaction.
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
            throw IOException(
                "Failed to create Lance table via namespace: " +
                (create_error.empty() ? "unknown error" : create_error));
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

          const char *data_storage_version_ptr =
              state->data_storage_version.empty()
                  ? nullptr
                  : state->data_storage_version.c_str();
          auto blob_flags = LanceBlobFlagsForTypes(state->column_types);
          state->writer = lance_open_writer_with_storage_options(
              state->open_path.c_str(), state->writer_mode.c_str(),
              key_ptrs.empty() ? nullptr : key_ptrs.data(),
              value_ptrs.empty() ? nullptr : value_ptrs.data(),
              state->option_keys.size(), LANCE_DEFAULT_MAX_ROWS_PER_FILE,
              LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
              LANCE_DEFAULT_MAX_BYTES_PER_FILE, data_storage_version_ptr,
              LanceGetSessionHandle(context), &state->schema_root.arrow_schema,
              blob_flags.empty() ? nullptr : blob_flags.data(),
              blob_flags.size(), LanceBlobIngestBatchBudget(context));
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

          // Resolve and validate source access before exporting the Arrow
          // array so a policy/secret failure cannot leak it.
          LanceBatchSourceOptions source_options;
          LanceCollectIngestSourceOptions(context.client, chunk,
                                          source_options);

          ArrowArray array;
          memset(&array, 0, sizeof(array));
          ArrowConverter::ToArrowArray(chunk, &array, props,
                                       extension_type_cast);

          auto rc = LanceWriteBlobBatch(gstate.writer, &array, source_options);
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
          chunk.data[0].Append(
              Value::BIGINT(NumericCast<int64_t>(gstate.insert_count)));
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
        string data_storage_version;
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

      string existing_id;
      auto exists = FindRestNamespaceTableId(
          rest_ns->namespace_id, create_info.GetTableName().GetIdentifierName(),
          rest_ns->delimiter, discovered, existing_id);

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
          rest_ns->api_key_override, rest_ns->headers_tsv,
          create_info.GetTableName().GetIdentifierName(), mode,
          data_storage_version, std::move(names), std::move(types),
          op.estimated_cardinality);
      create_as.children.push_back(plan);
      return create_as;
    }

    if (!IsSafeDatasetTableName(
            create_info.GetTableName().GetIdentifierName())) {
      throw InvalidInputException(
          "Unsafe Lance dataset name for CREATE TABLE: " +
          create_info.GetTableName().GetIdentifierName());
    }
    if (!directory_ns || directory_ns->root.empty()) {
      throw InternalException("Lance directory namespace root is empty");
    }

    auto physical_table = FindDirectoryNamespaceTable(
        *directory_ns, create_info.GetTableName().GetIdentifierName());
    auto exists = !physical_table.empty();
    if (!exists) {
      physical_table = create_info.GetTableName().GetIdentifierName();
    }
    auto dataset_path = JoinNamespacePath(directory_ns->root,
                                          GetDatasetDirName(physical_table));

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
    if (!data_storage_version.empty()) {
      copy_info.options["data_storage_version"] = {Value(data_storage_version)};
    }

    return PlanLanceCopy(
        context, planner, copy_info,
        StringsToIdentifiers(create_info.columns.GetColumnNames()),
        create_info.columns.GetColumnTypes(), plan);
  }

  // Lance plans DELETE before the child plan exists, so a plain
  // `DELETE FROM t` can drop whole fragments instead of scanning every row id.
  PhysicalOperator &PlanDelete(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalDelete &op) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceDelete(context, planner, op, nullptr);
    }
    return Catalog::PlanDelete(context, planner, op);
  }

  void ReplaceDefaultSchemaWithLanceSchema(CatalogTransaction transaction) {
    auto &schemas = GetSchemaCatalogSet();
    (void)schemas.DropEntry(transaction, DEFAULT_SCHEMA, true, true);

    CreateSchemaInfo info;
    info.SetSchema(DEFAULT_SCHEMA);
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    LogicalDependencyList dependencies;
    auto entry =
        make_uniq<LanceSchemaEntry>(*this, info, directory_ns, rest_ns);
    if (!schemas.CreateEntry(transaction, info.SchemaName(), std::move(entry),
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
  return PlanLanceDelete(context, planner, op, &plan);
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

  unique_ptr<LanceTableGenerator> generator;
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;

  auto is_rest_namespace = !endpoint.empty();
  string namespace_id;
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
    ResolveLanceNamespaceAuthOverrides(info.options, bearer_token_override,
                                       api_key_override);
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

  if (!is_rest_namespace) {
    generator = make_uniq<LanceDirectoryDefaultGenerator>(*catalog, schema,
                                                          directory_ns);
  } else {
    generator = make_uniq<LanceRestNamespaceDefaultGenerator>(
        *catalog, schema, endpoint, namespace_id, delimiter,
        bearer_token_override, api_key_override, headers_tsv);
  }
  lance_schema.SetTableDefaultGenerator(std::move(generator));

  (void)name;
  return std::move(catalog);
}

struct LancePendingAppend {
  string path;
  vector<string> option_keys;
  vector<string> option_values;
  string cache_key;
  void *transaction = nullptr;
  shared_ptr<LanceDatasetCacheEntry> pinned_dataset;
};

class LanceTransactionManager final : public DuckTransactionManager {
public:
  explicit LanceTransactionManager(AttachedDatabase &db)
      : DuckTransactionManager(db) {}

  void RegisterPendingAppend(Transaction &transaction_p,
                             LancePendingAppend pending) {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    lock_guard<mutex> guard(pending_lock);
    pending_appends[transaction.GetTransactionId()].push_back(
        std::move(pending));
  }

  bool HasPendingAppends() {
    lock_guard<mutex> guard(pending_lock);
    return !pending_appends.empty();
  }

  ErrorData CommitTransaction(ClientContext &context,
                              Transaction &transaction_p) override {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    vector<LancePendingAppend> appends;
    {
      lock_guard<mutex> guard(pending_lock);
      auto it = pending_appends.find(transaction.GetTransactionId());
      if (it != pending_appends.end()) {
        appends = std::move(it->second);
        pending_appends.erase(it);
      }
    }

    for (idx_t append_idx = 0; append_idx < appends.size(); append_idx++) {
      auto &pending = appends[append_idx];
      vector<const char *> key_ptrs;
      vector<const char *> value_ptrs;
      BuildStorageOptionPointerArrays(
          pending.option_keys, pending.option_values, key_ptrs, value_ptrs);

      auto rc = pending.pinned_dataset
                    ? lance_commit_transaction_to_dataset(
                          pending.pinned_dataset->Handle(), pending.transaction)
                    : lance_commit_transaction_with_storage_options(
                          pending.path.c_str(),
                          key_ptrs.empty() ? nullptr : key_ptrs.data(),
                          value_ptrs.empty() ? nullptr : value_ptrs.data(),
                          pending.option_keys.size(),
                          LanceGetSessionHandle(context), pending.transaction);
      if (rc != 0) {
        // Best-effort cleanup of any remaining pending transactions.
        // Note: the transaction pointer is consumed by the commit call, even on
        // error.
        for (idx_t cleanup_idx = append_idx + 1; cleanup_idx < appends.size();
             cleanup_idx++) {
          lance_free_transaction(appends[cleanup_idx].transaction);
        }
        DuckTransactionManager::RollbackTransaction(transaction_p);
        return ErrorData(ExceptionType::TRANSACTION,
                         "Failed to commit Lance append transaction for '" +
                             pending.path + "'" + LanceFormatErrorSuffix());
      }
    }

    auto result =
        DuckTransactionManager::CommitTransaction(context, transaction_p);
    if (!result.HasError() && !appends.empty()) {
      for (auto &pending : appends) {
        LanceInvalidateDatasetCache(context, pending.cache_key);
      }
    }
    return result;
  }

  void RollbackTransaction(Transaction &transaction_p) override {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    vector<LancePendingAppend> appends;
    {
      lock_guard<mutex> guard(pending_lock);
      auto it = pending_appends.find(transaction.GetTransactionId());
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

void RegisterLancePendingAppend(
    ClientContext &context, Catalog &catalog, string dataset_uri,
    vector<string> option_keys, vector<string> option_values, string cache_key,
    void *lance_transaction,
    shared_ptr<LanceDatasetCacheEntry> pinned_dataset) {
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
  pending.cache_key = std::move(cache_key);
  pending.transaction = lance_transaction;
  pending.pinned_dataset = std::move(pinned_dataset);
  tm->RegisterPendingAppend(txn, std::move(pending));
}

// Checks every attached Lance catalog, not only the one a caller targets: the
// same dataset can be reached through a catalog name and through a path, so a
// match on the dataset URI could miss it.
bool LanceHasUncommittedWrites(ClientContext &context) {
  for (auto &db : DatabaseManager::Get(context).GetDatabases(context)) {
    auto *tm =
        dynamic_cast<LanceTransactionManager *>(&db->GetTransactionManager());
    if (tm && tm->HasPendingAppends()) {
      return true;
    }
  }
  return false;
}

} // namespace duckdb
