#pragma once

#include "duckdb.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/storage/object_cache.hpp"

namespace duckdb {

class LanceSharedSessionEntry final : public ObjectCacheEntry {
public:
  LanceSharedSessionEntry();
  ~LanceSharedSessionEntry() override;

  static string ObjectType();
  string GetObjectType() override;
  optional_idx GetEstimatedCacheMemory() const override;

  void *Handle() const { return session; }
  uint64_t Id() const { return session_id; }

private:
  void *session = nullptr;
  uint64_t session_id = 0;
};

class LanceSessionState final : public ClientContextState {
public:
  explicit LanceSessionState(ClientContext &context);
  ~LanceSessionState() override;
  void WriteProfilingInformation(std::ostream &ss) override;

  void *Handle() const;

private:
  bool shared_session_cache_hit = false;
  shared_ptr<LanceSharedSessionEntry> shared_session;
};

shared_ptr<LanceSharedSessionEntry>
GetOrCreateLanceSharedSessionEntry(ClientContext &context,
                                   bool *out_cache_hit = nullptr);
shared_ptr<LanceSessionState>
GetOrCreateLanceSessionState(ClientContext &context);
void *LanceGetSessionHandle(ClientContext &context);

} // namespace duckdb
