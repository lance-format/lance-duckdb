#pragma once

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/table_storage_info.hpp"

namespace duckdb {

struct AlterInfo;
struct CatalogTransaction;
class CatalogEntry;
class ClientContext;

// LanceTableEntry represents a Lance dataset as a DuckDB base table entry.
// It supports scanning via a Lance-backed table scan function and appending via
// DuckDB's INSERT planning path (implemented at the catalog level).
class LanceTableEntry final : public TableCatalogEntry {
public:
  LanceTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                  CreateTableInfo &info, string dataset_uri);

  unique_ptr<CatalogEntry> AlterEntry(ClientContext &context,
                                      AlterInfo &info) override;
  unique_ptr<CatalogEntry> AlterEntry(CatalogTransaction transaction,
                                      AlterInfo &info) override;

  unique_ptr<CatalogEntry> Copy(ClientContext &context) const override;

  TableFunction GetScanFunction(ClientContext &context,
                                unique_ptr<FunctionData> &bind_data) override;

  unique_ptr<BaseStatistics> GetStatistics(ClientContext &, column_t) override {
    return nullptr;
  }

  TableStorageInfo GetStorageInfo(ClientContext &) override { return {}; }

  const string &DatasetUri() const { return dataset_uri; }

private:
  string dataset_uri;
};

} // namespace duckdb
