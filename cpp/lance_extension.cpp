#define DUCKDB_EXTENSION_MAIN

#include "lance_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"
#include "duckdb/main/config.hpp"

// Forward declarations for functions defined in other files
namespace duckdb {
    void RegisterLanceScan(ExtensionLoader &loader);
    void RegisterLanceCopy(ExtensionLoader &loader);
    void RegisterLanceReplacement(DBConfig &config);
}

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
    // Register table function for lance_scan
    RegisterLanceScan(loader);
    
    // Register copy function for COPY TO/FROM
    RegisterLanceCopy(loader);
}

void LanceExtension::Load(ExtensionLoader &loader) {
    LoadInternal(loader);
    
    // Register replacement scan
    auto &instance = loader.GetDatabaseInstance();
    auto &config = DBConfig::GetConfig(instance);
    RegisterLanceReplacement(config);
}

std::string LanceExtension::Name() {
    return "lance";
}

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
