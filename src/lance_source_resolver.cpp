#include "lance_source_resolver.hpp"

#include "lance_common.hpp"
#include "lance_session_state.hpp"

#include "duckdb/main/client_data.hpp"

namespace duckdb {

struct LanceSourceCallbackResult {
  vector<string> keys;
  vector<string> values;
  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  string error;
};

static thread_local LanceSourceCallbackResult SOURCE_CALLBACK_RESULT;

LanceSourceResolver::LanceSourceResolver(ClientContext &context)
    : context(context.shared_from_this()) {}

bool LanceSourceResolver::ResolveSource(const string &uri,
                                        vector<string> &out_keys,
                                        vector<string> &out_values,
                                        string &out_error) const {
  auto locked_context = context.lock();
  if (!locked_context) {
    out_error = "DuckDB connection for source resolution is closed";
    return false;
  }
  auto &config = DBConfig::GetConfig(*locked_context);
  if (!config.CanAccessFile(
          uri, FileType::FILE_TYPE_REGULAR,
          ClientData::Get(*locked_context).file_opener.get())) {
    out_error = "Cannot access source URI \"" + uri +
                "\" - file system operations are disabled by configuration";
    return false;
  }
  // This is the same context transaction and SecretManager path used by
  // DuckDB's KeyValueSecretReader(FileOpener &) on file-I/O worker threads.
  LanceFillStorageOptionsFromSecrets(*locked_context, uri, out_keys,
                                     out_values);
  return true;
}

shared_ptr<LanceSourceResolver>
LanceCreateSourceResolver(ClientContext &context) {
  return make_shared_ptr<LanceSourceResolver>(context);
}

void *
LanceCreateSourceSession(ClientContext &context,
                         const shared_ptr<LanceSourceResolver> &resolver) {
  if (!resolver) {
    return nullptr;
  }
  auto *owned = new LanceSourceResolver(*resolver);
  auto callbacks = LanceBuildSourceResolverCallbacks(owned);
  auto *session =
      lance_create_source_session(LanceGetSessionHandle(context), &callbacks);
  if (!session) {
    throw IOException("Failed to create Lance source session" +
                      LanceFormatErrorSuffix());
  }
  return session;
}

void *LanceOpenDatasetWithSourceSession(
    ClientContext &context, const shared_ptr<LanceSourceResolver> &resolver,
    const std::function<void *(void *session)> &open) {
  auto *session = LanceCreateSourceSession(context, resolver);
  try {
    auto *dataset = open(session);
    lance_close_session(session);
    return dataset;
  } catch (...) {
    lance_close_session(session);
    throw;
  }
}

static int32_t ResolveSourceCallback(void *opaque, const char *uri,
                                     const char ***out_keys,
                                     const char ***out_values,
                                     size_t *out_count,
                                     const char **out_error) {
  if (!out_keys || !out_values || !out_count || !out_error) {
    return -1;
  }
  *out_keys = nullptr;
  *out_values = nullptr;
  *out_count = 0;
  *out_error = nullptr;
  if (!opaque || !uri) {
    SOURCE_CALLBACK_RESULT.error = "Invalid source resolver callback input";
    *out_error = SOURCE_CALLBACK_RESULT.error.c_str();
    return -1;
  }

  try {
    SOURCE_CALLBACK_RESULT = LanceSourceCallbackResult();
    if (!static_cast<LanceSourceResolver *>(opaque)->ResolveSource(
            uri, SOURCE_CALLBACK_RESULT.keys, SOURCE_CALLBACK_RESULT.values,
            SOURCE_CALLBACK_RESULT.error)) {
      *out_error = SOURCE_CALLBACK_RESULT.error.c_str();
      return -1;
    }
    if (SOURCE_CALLBACK_RESULT.keys.size() !=
        SOURCE_CALLBACK_RESULT.values.size()) {
      SOURCE_CALLBACK_RESULT.error =
          "Lance source storage option keys/values size mismatch";
      *out_error = SOURCE_CALLBACK_RESULT.error.c_str();
      return -1;
    }
    SOURCE_CALLBACK_RESULT.key_ptrs.reserve(SOURCE_CALLBACK_RESULT.keys.size());
    SOURCE_CALLBACK_RESULT.value_ptrs.reserve(
        SOURCE_CALLBACK_RESULT.values.size());
    for (auto &key : SOURCE_CALLBACK_RESULT.keys) {
      SOURCE_CALLBACK_RESULT.key_ptrs.push_back(key.c_str());
    }
    for (auto &value : SOURCE_CALLBACK_RESULT.values) {
      SOURCE_CALLBACK_RESULT.value_ptrs.push_back(value.c_str());
    }
    *out_count = SOURCE_CALLBACK_RESULT.keys.size();
    if (*out_count > 0) {
      *out_keys = SOURCE_CALLBACK_RESULT.key_ptrs.data();
      *out_values = SOURCE_CALLBACK_RESULT.value_ptrs.data();
    }
    return 0;
  } catch (std::exception &ex) {
    SOURCE_CALLBACK_RESULT.error = ex.what();
    *out_error = SOURCE_CALLBACK_RESULT.error.c_str();
    return -1;
  }
}

static void DestroySourceResolverCallback(void *opaque) {
  delete static_cast<LanceSourceResolver *>(opaque);
}

static void ReleaseSourceCallback(void *) {
  SOURCE_CALLBACK_RESULT = LanceSourceCallbackResult();
}

LanceSourceResolverCallbacks
LanceBuildSourceResolverCallbacks(LanceSourceResolver *resolver) {
  return LanceSourceResolverCallbacks{
      resolver,
      ResolveSourceCallback,
      ReleaseSourceCallback,
      DestroySourceResolverCallback,
  };
}

} // namespace duckdb
