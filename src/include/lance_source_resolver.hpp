#pragma once

#include "duckdb.hpp"
#include "lance_ffi.hpp"

#include <functional>
namespace duckdb {

// A runtime-only handle into the opening ClientContext.
//
// The weak reference avoids a DatasetCache -> resolver -> ClientContext
// ownership cycle while still allowing provider callbacks to reuse DuckDB's
// native secret lookup and file-access checks for the actual source URI.
class LanceSourceResolver {
public:
  explicit LanceSourceResolver(ClientContext &context);

  bool ResolveSource(const string &uri, vector<string> &out_keys,
                     vector<string> &out_values, string &out_error) const;

private:
  weak_ptr<ClientContext> context;
};

shared_ptr<LanceSourceResolver>
LanceCreateSourceResolver(ClientContext &context);

// Create a Lance session that shares the database-wide cache backends but has
// an isolated object-store registry for this resolver. The returned handle must
// be closed after the dataset is opened; the Dataset keeps its Arc<Session>
// alive. Closing a derived handle does not clear the shared caches.
void *LanceCreateSourceSession(ClientContext &context,
                               const shared_ptr<LanceSourceResolver> &resolver);

// Run one dataset open on a derived source session. The Dataset returned by
// `open` retains the session, so the temporary FFI handle is closed here.
void *LanceOpenDatasetWithSourceSession(
    ClientContext &context, const shared_ptr<LanceSourceResolver> &resolver,
    const std::function<void *(void *session)> &open);

LanceSourceResolverCallbacks
LanceBuildSourceResolverCallbacks(LanceSourceResolver *resolver);

} // namespace duckdb
