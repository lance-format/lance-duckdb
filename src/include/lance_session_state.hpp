#pragma once

#include "duckdb.hpp"
#include "duckdb/main/client_context_state.hpp"

namespace duckdb {

class LanceSessionState final : public ClientContextState {
public:
  LanceSessionState();
  ~LanceSessionState() override;
  void WriteProfilingInformation(std::ostream &ss) override;

  void *Handle() const { return session; }

private:
  void *session = nullptr;
};

shared_ptr<LanceSessionState>
GetOrCreateLanceSessionState(ClientContext &context);
void *LanceGetSessionHandle(ClientContext &context);

} // namespace duckdb
