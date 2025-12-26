#define DUCKDB_EXTENSION_MAIN

#include "lance_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// Forward declaration
void RegisterLanceScan(ExtensionLoader &loader);
void RegisterLanceSearch(ExtensionLoader &loader);
void RegisterLanceReplacement(DBConfig &config);
void RegisterLanceWrite(ExtensionLoader &loader);
void RegisterLanceStorage(DBConfig &config);

static void LoadInternal(ExtensionLoader &loader) {
  // Register the lance_scan table function
  RegisterLanceScan(loader);
  RegisterLanceSearch(loader);
  RegisterLanceWrite(loader);
}

void LanceExtension::Load(ExtensionLoader &loader) {
  LoadInternal(loader);

  // Enable SELECT * FROM '.../dataset.lance'
  auto &instance = loader.GetDatabaseInstance();
  auto &config = DBConfig::GetConfig(instance);
  RegisterLanceStorage(config);
  RegisterLanceReplacement(config);
}

std::string LanceExtension::Name() { return "lance"; }

std::string LanceExtension::Version() const {
#ifdef EXT_VERSION_LANCE
  return EXT_VERSION_LANCE;
#else
  return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(lance, loader) {
  duckdb::LanceExtension extension;
  extension.Load(loader);
}

DUCKDB_EXTENSION_API void lance_init(duckdb::DatabaseInstance &db) {
  duckdb::DuckDB db_wrapper(db);
  db_wrapper.LoadStaticExtension<duckdb::LanceExtension>();
}

DUCKDB_EXTENSION_API const char *lance_version() {
  return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
