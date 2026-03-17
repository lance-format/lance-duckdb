#define DUCKDB_EXTENSION_MAIN

#include "lance_extension.hpp"
#include "lance_secrets.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// Forward declaration
void RegisterLanceMaintenance(ExtensionLoader &loader);
void RegisterLanceMaintenanceParser(DBConfig &config);
void RegisterLanceScan(ExtensionLoader &loader);
void RegisterLanceSearch(ExtensionLoader &loader);
void RegisterLanceReplacement(DBConfig &config);
void RegisterLanceWrite(ExtensionLoader &loader);
void RegisterLanceStorage(DBConfig &config);
void RegisterLanceTruncate(DBConfig &config, ExtensionLoader &loader);
void RegisterLanceIndex(DBConfig &config, ExtensionLoader &loader);
void RegisterLanceScanOptimizer(DBConfig &config);

static void LoadInternal(ExtensionLoader &loader) {
  // Register internal scan table functions.
  RegisterLanceScan(loader);
  RegisterLanceSearch(loader);
  RegisterLanceWrite(loader);
  RegisterLanceMaintenance(loader);
}

void LanceExtension::Load(ExtensionLoader &loader) {
  LoadInternal(loader);

  // Enable SELECT * FROM '.../dataset.lance'
  auto &instance = loader.GetDatabaseInstance();
  RegisterLanceSecrets(instance);
  auto &config = DBConfig::GetConfig(instance);
  config.AddExtensionOption("lance_deferred_materialization",
                            "Enable deferred materialization for heavy columns "
                            "when filter pushdown fails",
                            LogicalType::BOOLEAN, Value::BOOLEAN(true));
  RegisterLanceScanOptimizer(config);
  RegisterLanceStorage(config);
  RegisterLanceReplacement(config);
  RegisterLanceTruncate(config, loader);
  RegisterLanceIndex(config, loader);
  RegisterLanceMaintenanceParser(config);
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
