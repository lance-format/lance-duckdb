#pragma once

#include "duckdb.hpp"

namespace duckdb {

string LanceConsumeLastError();
string LanceFormatErrorSuffix();

bool IsComputedSearchColumn(const string &name);

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel);

void *LanceOpenDataset(ClientContext &context, const string &path);

string LanceNormalizeS3Scheme(const string &path);
void LanceFillS3StorageOptionsFromSecrets(ClientContext &context,
                                          const string &path,
                                          vector<string> &out_keys,
                                          vector<string> &out_values);

} // namespace duckdb
