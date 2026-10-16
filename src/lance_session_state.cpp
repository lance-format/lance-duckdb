#include "lance_session_state.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"

#include <atomic>

namespace duckdb {

static constexpr const char *LANCE_SESSION_STATE_KEY = "lance_session_state";
static constexpr const char *LANCE_SHARED_SESSION_CACHE_KEY =
    "lance.shared_session.v1";
static std::atomic<uint64_t> LANCE_SHARED_SESSION_ID_GEN{1};

LanceSharedSessionEntry::LanceSharedSessionEntry() {
  session_id = LANCE_SHARED_SESSION_ID_GEN.fetch_add(1);
  session = lance_create_session(0, 0);
  if (!session) {
    throw IOException("Failed to create Lance session" +
                      LanceFormatErrorSuffix());
  }
}

LanceSharedSessionEntry::~LanceSharedSessionEntry() {
  if (session) {
    lance_close_session(session);
    session = nullptr;
  }
}

string LanceSharedSessionEntry::ObjectType() { return "lance_shared_session"; }

string LanceSharedSessionEntry::GetObjectType() { return ObjectType(); }

optional_idx LanceSharedSessionEntry::GetEstimatedCacheMemory() const {
  return optional_idx{};
}

shared_ptr<LanceSharedSessionEntry>
GetOrCreateLanceSharedSessionEntry(ClientContext &context,
                                   bool *out_cache_hit) {
  auto &cache = ObjectCache::GetObjectCache(context);
  auto existing =
      cache.Get<LanceSharedSessionEntry>(LANCE_SHARED_SESSION_CACHE_KEY);
  if (out_cache_hit) {
    *out_cache_hit = existing != nullptr;
  }
  auto entry = existing ? existing
                        : cache.GetOrCreate<LanceSharedSessionEntry>(
                              LANCE_SHARED_SESSION_CACHE_KEY);
  if (!entry || !entry->Handle()) {
    throw IOException("Failed to access shared Lance session");
  }
  return entry;
}

LanceSessionState::LanceSessionState(ClientContext &context)
    : shared_session(GetOrCreateLanceSharedSessionEntry(
          context, &shared_session_cache_hit)) {}

LanceSessionState::~LanceSessionState() = default;

void *LanceSessionState::Handle() const {
  return shared_session ? shared_session->Handle() : nullptr;
}

void LanceSessionState::WriteProfilingInformation(std::ostream &ss) {
  LanceSessionStats stats{};

  auto *session = Handle();
  if (session && lance_session_get_stats(session, &stats) == 0) {
    ss << "Lance Session Cache: scope=database_shared shared_session_id="
       << shared_session->Id()
       << " object_cache_hit=" << (shared_session_cache_hit ? "true" : "false")
       << " approx_num_items=" << stats.approx_num_items
       << " size_bytes=" << stats.size_bytes << "\n";
    return;
  }

  ss << "Lance Session Cache: scope=database_shared shared_session_id="
     << (shared_session ? shared_session->Id() : 0)
     << " object_cache_hit=" << (shared_session_cache_hit ? "true" : "false")
     << " unavailable\n";
}

shared_ptr<LanceSessionState>
GetOrCreateLanceSessionState(ClientContext &context) {
  return context.registered_state->GetOrCreate<LanceSessionState>(
      LANCE_SESSION_STATE_KEY, context);
}

void *LanceGetSessionHandle(ClientContext &context) {
  auto state = GetOrCreateLanceSessionState(context);
  if (!state || !state->Handle()) {
    throw IOException("Failed to access Lance session state");
  }
  return state->Handle();
}

} // namespace duckdb
