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
void RegisterLanceKNNScan(ExtensionLoader &loader);
void RegisterLanceKNNOptimizer(DBConfig &config);

static void LoadInternal(ExtensionLoader &loader) {
  // Register internal scan table functions.
  RegisterLanceScan(loader);
  RegisterLanceSearch(loader);
  RegisterLanceWrite(loader);
  RegisterLanceMaintenance(loader);
  RegisterLanceKNNScan(loader);
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
  config.AddExtensionOption(
      "lance_knn_nprobes",
      "Default nprobes passed to Lance KNN scans rewritten by the optimizer "
      "(0 = Lance default)",
      LogicalType::BIGINT, Value::BIGINT(0));
  config.AddExtensionOption(
      "lance_knn_refine_factor",
      "Default refine_factor passed to Lance KNN scans rewritten by the "
      "optimizer (0 = no refinement)",
      LogicalType::BIGINT, Value::BIGINT(0));
  config.AddExtensionOption(
      "lance_knn_ef_search",
      "HNSW search-time candidate-pool size (ef_search) passed to Lance KNN "
      "scans rewritten by the optimizer. Larger values trade latency for "
      "recall; 0 = Lance default (k + k/2). Distinct from hnsw_ef_construction "
      "which only affects index build.",
      LogicalType::BIGINT, Value::BIGINT(0));
  config.AddExtensionOption(
      "lance_knn_prefilter",
      "Whether the optimizer-rewritten Lance KNN scans should pre-filter "
      "(true) or leave filtered queries on DuckDB's original TopN plan "
      "(false). Explicit lance_vector_search(prefilter=false) remains "
      "available for post-filtered KNN search.",
      LogicalType::BOOLEAN, Value::BOOLEAN(true));
  config.AddExtensionOption(
      "lance_knn_use_index",
      "Kill switch for the Lance KNN optimizer — when false the optimizer "
      "leaves ORDER BY <distance> LIMIT k plans alone and DuckDB executes "
      "them via a full Lance scan + in-memory top-k",
      LogicalType::BOOLEAN, Value::BOOLEAN(true));
  config.AddExtensionOption(
      "lance_knn_max_k",
      "Upper bound on the LIMIT value the Lance KNN optimizer will rewrite. "
      "Plans with LIMIT above this value are left to DuckDB so a runaway "
      "query does not pin an arbitrarily large k inside the index scan. "
      "Setting to 0 falls back to the built-in default (10000); the "
      "optimizer also clamps any configured value to a hard ceiling of "
      "1000000 to keep a typo'd setting from blowing up memory",
      LogicalType::BIGINT, Value::BIGINT(10000));
  RegisterLanceScanOptimizer(config);
  RegisterLanceKNNOptimizer(config);
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
