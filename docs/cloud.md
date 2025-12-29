# Cloud Storage

This extension can read and write Lance datasets stored in object stores by passing an object-store URI (e.g. `s3://...`)
to DuckDB, and configuring credentials and client settings via DuckDB's Secret Manager.

This integration is intentionally thin:

- You configure credentials and object store settings using `CREATE SECRET (TYPE LANCE, ...)`.
- The extension reads the matching secret by URI prefix (`SCOPE`) and forwards the key/value pairs to Lance as
  `storage_options`.

For the upstream Lance object store options and provider behavior, see https://lance.org/guide/object_store/.

## Supported Backends

Backends are supported if they are supported by the linked Lance build (see `Cargo.toml` features).

Common URI schemes:

- S3 / S3-compatible: `s3://...` (also accepts `s3a://...` and `s3n://...`, normalized to `s3://...`)
- Google Cloud Storage: `gs://...`
- Azure Blob Storage: `az://...`
- Alibaba Cloud OSS: `oss://...`
- Hugging Face Hub (OpenDAL): `hf://...`
- Local filesystem paths: `path/to/dataset.lance`

## Secret Type: `LANCE`

The extension registers a dedicated secret type with multiple providers:

- `PROVIDER config`: Explicit key/value configuration in SQL.
- `PROVIDER credential_chain`: Create a scoped secret but rely on the upstream SDK credential chain (env vars, shared
  config/credentials profiles, instance metadata) for credentials.
- `PROVIDER env`: Create a scoped secret but rely on environment variables in the upstream provider/SDK.

All providers forward the resulting key/value pairs to Lance as `storage_options`.

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER config,
  SCOPE 's3://my-bucket/',
  ACCESS_KEY_ID '...',
  SECRET_ACCESS_KEY '...',
  REGION 'us-east-1'
);
```

### Scope Matching

Secrets are matched using DuckDB's Secret Manager prefix matching:

- `SCOPE` should be a URI prefix (e.g. `s3://bucket/`, `gs://bucket/`, `az://container/`).
- The best (longest) matching prefix is used when opening a dataset.

### Parameter Binding and Forwarding

DuckDB validates `CREATE SECRET` parameters at bind time. For `TYPE LANCE`, the extension maintains an allowlist of
known Lance storage option keys. Values are stored as strings and forwarded to Lance as-is.

If you need to pass keys that are not in the allowlist yet, you can use the fallback `STORAGE_OPTIONS` map:

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER config,
  SCOPE 's3://my-bucket/',
  STORAGE_OPTIONS map(['my_custom_key'], ['my_custom_value'])
);
```

### Using the Credential Chain

For object stores that have an upstream SDK credential chain (e.g. AWS), you can create a scoped secret without
embedding credentials in DuckDB:

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER credential_chain,
  SCOPE 's3://my-bucket/',
  REGION 'us-east-1'
);
```

## Common Options (Any Backend)

These keys are commonly used across providers:

- `ALLOW_HTTP`: Allow HTTP (insecure) connections (useful for MinIO / local testing).
- `ALLOW_INVALID_CERTIFICATES`: Allow invalid TLS certificates (use with care).
- `CONNECT_TIMEOUT`: Connection timeout (string value forwarded to Lance).
- `REQUEST_TIMEOUT`: Request timeout (string value forwarded to Lance).
- `CLIENT_MAX_RETRIES`: Retry count for object store client.
- `CLIENT_RETRY_TIMEOUT`: Retry timeout for object store client.
- `DOWNLOAD_RETRY_COUNT`: Download retry count for reads.
- `PROXY_URL`: Proxy URL.
- `PROXY_CA_CERTIFICATE`: Proxy CA certificate path or content (provider dependent).
- `PROXY_EXCLUDES`: Comma-separated proxy bypass list.
- `USER_AGENT`: Custom user agent.
- `EXPIRES_AT_MILLIS`: Optional expiration timestamp (milliseconds since epoch) for short-lived credentials.
- `USE_OPENDAL`: Use OpenDAL-based provider implementation when available (provider dependent).

## S3 / S3-Compatible (`s3://`)

Typical keys:

- `ACCESS_KEY_ID`: AWS access key id.
- `SECRET_ACCESS_KEY`: AWS secret access key.
- `SESSION_TOKEN`: AWS session token (optional).
- `REGION`: AWS region.
- `ENDPOINT`: Custom endpoint (MinIO, localstack, non-AWS S3-compatible).
- `VIRTUAL_HOSTED_STYLE_REQUEST`: `true`/`false`.
- `SKIP_SIGNATURE`: `true` to use unsigned requests (anonymous / public buckets).

The extension also accepts `AWS_*` aliases (e.g. `AWS_ACCESS_KEY_ID`, `AWS_REGION`, ...).

Note: DuckDB requires `INSTALL httpfs;` and `LOAD httpfs;` before running
`ATTACH 's3://...' AS ... (TYPE LANCE, ...)`.

Example (MinIO):

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER config,
  SCOPE 's3://my-bucket/',
  ACCESS_KEY_ID 'minioadmin',
  SECRET_ACCESS_KEY 'minioadmin',
  REGION 'us-east-1',
  ENDPOINT 'http://127.0.0.1:9000',
  VIRTUAL_HOSTED_STYLE_REQUEST false,
  ALLOW_HTTP true
);
```

## Google Cloud Storage (`gs://`)

Typical keys:

- `SERVICE_ACCOUNT`: Service account email (OpenDAL, provider dependent).
- `SERVICE_ACCOUNT_KEY`: Service account key (JSON, provider dependent).
- `APPLICATION_CREDENTIALS`: Application credentials (provider dependent).
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to credentials file (provider dependent).
- `GOOGLE_SERVICE_ACCOUNT`: Service account email (provider dependent).
- `GOOGLE_SERVICE_ACCOUNT_KEY`: Service account key (provider dependent).
- `GOOGLE_STORAGE_TOKEN`: Bearer token for GCS requests (when using object_store's GCS client).
- `USE_OPENDAL`: `true`/`false` to select OpenDAL implementation.

Example (token-based):

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER config,
  SCOPE 'gs://my-bucket/',
  GOOGLE_STORAGE_TOKEN '...',
  USE_OPENDAL false
);
```

## Azure Blob Storage (`az://`)

Typical keys:

- `ACCOUNT_NAME`: Storage account name.
- `ACCOUNT_KEY`: Storage account key (base64).
- `SAS_KEY` / `SAS_TOKEN`: SAS credentials (provider dependent).
- `BEARER_TOKEN`: Bearer token (provider dependent).
- `AZURE_*`: Additional Azure configuration keys forwarded to Lance (see Lance docs).
- `USE_OPENDAL`: `true`/`false` to select OpenDAL implementation.

Note: Depending on the URI form, the account name may need to be present in either the URI authority or in the options.

Example (account key):

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER config,
  SCOPE 'az://my-container/',
  ACCOUNT_NAME 'my-account',
  ACCOUNT_KEY '...',
  USE_OPENDAL true
);
```

## Alibaba Cloud OSS (`oss://`)

Typical keys:

- `OSS_ENDPOINT`: Required OSS endpoint.
- `OSS_ACCESS_KEY_ID`: Access key id.
- `OSS_SECRET_ACCESS_KEY`: Access key secret.
- `OSS_REGION`: Region (optional, provider dependent).

Example:

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER config,
  SCOPE 'oss://my-bucket/',
  OSS_ENDPOINT 'https://oss-cn-hangzhou.aliyuncs.com',
  OSS_ACCESS_KEY_ID '...',
  OSS_SECRET_ACCESS_KEY '...'
);
```

## Hugging Face Hub (`hf://`)

Hugging Face support is implemented via OpenDAL. URLs look like:

- `hf://datasets/<owner>/<repo>/<path>`

Typical keys:

- `HF_TOKEN`: Access token.
- `HF_REVISION`: Revision name (e.g. `main`, tag, commit).
- `HF_ROOT`: Root prefix inside the repo.

Example:

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER config,
  SCOPE 'hf://datasets/acme/my-repo/',
  HF_TOKEN '...',
  HF_REVISION 'main'
);
```

## Notes on Secret Redaction

When listing secrets, DuckDB may redact values depending on secret display mode. Additionally, the `TYPE LANCE` secret
marks keys containing `secret`, `password`, or `token` as redacted by default.
