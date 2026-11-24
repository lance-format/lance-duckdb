#pragma once

namespace duckdb {
class Catalog;
class ClientContext;

bool IsLanceCatalog(Catalog &catalog);
void RequireLanceTableDDLAutocommit(ClientContext &context);
} // namespace duckdb
