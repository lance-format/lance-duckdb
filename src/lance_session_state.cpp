#include "lance_session_state.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"

#include "duckdb/main/client_context_state.hpp"

namespace duckdb {

static constexpr const char *LANCE_SESSION_STATE_KEY = "lance_session_state";

LanceSessionState::LanceSessionState() {
  session = lance_create_session(0, 0);
  if (!session) {
    throw IOException("Failed to create Lance session" +
                      LanceFormatErrorSuffix());
  }
}

LanceSessionState::~LanceSessionState() {
  if (session) {
    lance_close_session(session);
    session = nullptr;
  }
}

void LanceSessionState::WriteProfilingInformation(std::ostream &ss) {
  LanceSessionStats stats;
  memset(&stats, 0, sizeof(stats));
  if (session && lance_session_get_stats(session, &stats) == 0) {
    ss << "Lance Session Cache: approx_num_items=" << stats.approx_num_items
       << " size_bytes=" << stats.size_bytes << "\n";
    return;
  }
  ss << "Lance Session Cache: unavailable\n";
}
shared_ptr<LanceSessionState>
GetOrCreateLanceSessionState(ClientContext &context) {
  return context.registered_state->GetOrCreate<LanceSessionState>(
      LANCE_SESSION_STATE_KEY);
}

void *LanceGetSessionHandle(ClientContext &context) {
  auto state = GetOrCreateLanceSessionState(context);
  if (!state || !state->Handle()) {
    throw IOException("Failed to access Lance session state");
  }
  return state->Handle();
}

} // namespace duckdb
