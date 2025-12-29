#include "lance_secrets.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/main/secret/secret_manager.hpp"

namespace duckdb {

static bool ShouldRedactLanceStorageOption(const string &key) {
  return StringUtil::Contains(StringUtil::Lower(key), "secret") ||
         StringUtil::Contains(StringUtil::Lower(key), "password") ||
         StringUtil::Contains(StringUtil::Lower(key), "token");
}

static unique_ptr<BaseSecret>
CreateLanceSecretFromConfig(ClientContext &, CreateSecretInput &input) {
  auto secret = make_uniq<KeyValueSecret>(input.scope, input.type,
                                          input.provider, input.name);

  for (auto &kv : input.options) {
    if (StringUtil::CIEquals(kv.first, "storage_options")) {
      continue;
    }
    if (kv.second.IsNull()) {
      continue;
    }
    secret->secret_map[kv.first] = Value(kv.second.ToString());
  }

  auto it = input.options.find("storage_options");
  if (it != input.options.end() && !it->second.IsNull()) {
    auto &children = MapValue::GetChildren(it->second);
    for (auto &child : children) {
      auto &child_struct = StructValue::GetChildren(child);
      if (child_struct[0].IsNull() || child_struct[1].IsNull()) {
        continue;
      }
      auto key = child_struct[0].ToString();
      auto value = child_struct[1].ToString();
      if (key.empty()) {
        continue;
      }
      secret->secret_map[key] = Value(std::move(value));
    }
  }

  for (auto &kv : secret->secret_map) {
    if (ShouldRedactLanceStorageOption(kv.first)) {
      secret->redact_keys.insert(kv.first);
    }
  }

  return std::move(secret);
}

static bool HasSecretType(SecretManager &secret_manager, const string &name) {
  for (auto &t : secret_manager.AllSecretTypes()) {
    if (StringUtil::CIEquals(t.name, name)) {
      return true;
    }
  }
  return false;
}

void RegisterLanceSecrets(DatabaseInstance &instance) {
  auto &secret_manager = SecretManager::Get(instance);

  if (!HasSecretType(secret_manager, "lance")) {
    SecretType secret_type;
    secret_type.name = "lance";
    secret_type.deserializer = KeyValueSecret::Deserialize<KeyValueSecret>;
    secret_type.default_provider = "config";
    secret_type.extension = "lance";
    secret_manager.RegisterSecretType(secret_type);
  }

  const auto storage_option_keys = vector<string>{
      "access_key_id",
      "secret_access_key",
      "session_token",
      "region",
      "endpoint",
      "virtual_hosted_style_request",
      "skip_signature",
      "allow_http",
      "allow_invalid_certificates",
      "connect_timeout",
      "request_timeout",
      "client_max_retries",
      "client_retry_timeout",
      "download_retry_count",
      "proxy_url",
      "proxy_ca_certificate",
      "proxy_excludes",
      "user_agent",
      "expires_at_millis",
      "use_opendal",
      "s3_express",
      "aws_access_key_id",
      "aws_secret_access_key",
      "aws_session_token",
      "aws_region",
      "aws_endpoint",
      "aws_virtual_hosted_style_request",
      "aws_server_side_encryption",
      "aws_sse_kms_key_id",
      "aws_sse_bucket_key_enabled",
      "aws_s3_express",
      "google_storage_token",
      "service_account",
      "service_account_key",
      "application_credentials",
      "google_application_credentials",
      "google_service_account",
      "google_service_account_key",
      "account_name",
      "account_key",
      "sas_key",
      "sas_token",
      "bearer_token",
      "access_key",
      "azure_storage_account_name",
      "azure_storage_account_key",
      "azure_client_id",
      "azure_client_secret",
      "azure_tenant_id",
      "azure_storage_sas_key",
      "azure_storage_sas_token",
      "azure_storage_token",
      "azure_endpoint",
      "azure_identity_endpoint",
      "azure_msi_endpoint",
      "azure_msi_resource_id",
      "azure_object_id",
      "azure_federated_token_file",
      "azure_use_azure_cli",
      "azure_use_fabric_endpoint",
      "azure_disable_tagging",
      "azure_storage_use_emulator",
      "oss_endpoint",
      "oss_access_key_id",
      "oss_secret_access_key",
      "oss_region",
      "hf_token",
      "hf_revision",
      "hf_root",
  };

  // DuckDB's CREATE SECRET binder requires an explicit allowlist of named
  // parameters. Values are stored as strings and forwarded to Lance as-is.
  auto register_provider = [&](const string &provider) {
    CreateSecretFunction fun;
    fun.secret_type = "lance";
    fun.provider = provider;
    fun.function = CreateLanceSecretFromConfig;

    for (auto &key : storage_option_keys) {
      fun.named_parameters[key] = LogicalType::VARCHAR;
    }

    // Fallback: accept a MAP for bulk configuration.
    fun.named_parameters["storage_options"] =
        LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);

    secret_manager.RegisterSecretFunction(
        std::move(fun), OnCreateConflict::REPLACE_ON_CONFLICT);
  };

  // Explicit key/value configuration.
  register_provider("config");

  // Semantic aliases:
  // - "credential_chain": configure scope/overrides, but rely on the upstream
  //   SDK credential chain (env/profile/metadata services) for credentials.
  // - "env": configure via environment variables in the upstream SDK/provider.
  register_provider("credential_chain");
  register_provider("env");
}

} // namespace duckdb
