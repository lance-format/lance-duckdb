#pragma once

#include "duckdb/common/arrow/arrow.hpp"

#include <cstddef>
#include <cstdint>

extern "C" {
typedef struct LanceSessionStats {
  uint64_t size_bytes;
  uint64_t approx_num_items;
} LanceSessionStats;

typedef struct LanceDebugCounters {
  uint64_t dataset_open_count;
  uint64_t namespace_describe_count;
  uint64_t commit_count;
} LanceDebugCounters;

void *lance_create_session(uint64_t index_cache_size_bytes,
                           uint64_t metadata_cache_size_bytes);
void lance_close_session(void *session);
int32_t lance_session_get_stats(void *session, LanceSessionStats *out_stats);
int32_t lance_debug_get_counters(LanceDebugCounters *out_counters);
void lance_debug_reset_counters();

void *lance_open_dataset(const char *path);
void *lance_open_dataset_with_session(const char *path, void *session);
void *lance_open_dataset_with_storage_options(const char *path,
                                              const char **option_keys,
                                              const char **option_values,
                                              size_t options_len);
void *lance_open_dataset_with_storage_options_and_session(
    const char *path, const char **option_keys, const char **option_values,
    size_t options_len, void *session);
const char *lance_dir_namespace_list_tables(const char *root,
                                            const char **option_keys,
                                            const char **option_values,
                                            size_t options_len);
int32_t lance_dir_namespace_drop_table(const char *root, const char *table_name,
                                       const char **option_keys,
                                       const char **option_values,
                                       size_t options_len);
void *lance_open_dataset_in_dir_namespace(
    const char *root, const char *table_name, const char **option_keys,
    const char **option_values, size_t options_len, const char **out_table_uri);
void *lance_open_dataset_in_dir_namespace_with_session(
    const char *root, const char *table_name, const char **option_keys,
    const char **option_values, size_t options_len, void *session,
    const char **out_table_uri);
const char *
lance_namespace_list_tables(const char *endpoint, const char *namespace_id,
                            const char *bearer_token, const char *api_key,
                            const char *delimiter, const char *headers_tsv);
int32_t lance_json_arrow_schema_to_c(const char *json_schema,
                                     ArrowSchema *out_schema);
int32_t lance_namespace_describe_table_with_schema(
    const char *endpoint, const char *table_id, const char *bearer_token,
    const char *api_key, const char *delimiter, const char *headers_tsv,
    const char **out_schema_json);
int32_t lance_namespace_describe_table(
    const char *endpoint, const char *table_id, const char *bearer_token,
    const char *api_key, const char *delimiter, const char *headers_tsv,
    const char **out_location, const char **out_storage_options_tsv);
int32_t lance_namespace_create_empty_table(
    const char *endpoint, const char *table_id, const char *bearer_token,
    const char *api_key, const char *delimiter, const char *headers_tsv,
    const char **out_location, const char **out_storage_options_tsv);
int32_t lance_namespace_drop_table(const char *endpoint, const char *table_id,
                                   const char *bearer_token,
                                   const char *api_key, const char *delimiter,
                                   const char *headers_tsv);
void *
lance_open_dataset_in_namespace(const char *endpoint, const char *table_id,
                                const char *bearer_token, const char *api_key,
                                const char *delimiter, const char *headers_tsv,
                                const char **out_table_uri);
void *lance_open_dataset_in_namespace_with_session(
    const char *endpoint, const char *table_id, const char *bearer_token,
    const char *api_key, const char *delimiter, const char *headers_tsv,
    void *session, const char **out_table_uri);
void lance_close_dataset(void *dataset);

void *lance_get_schema(void *dataset);
void *lance_get_schema_for_scan(void *dataset);
void lance_free_schema(void *schema);
int32_t lance_schema_to_arrow(void *schema, ArrowSchema *out_schema);

int32_t lance_stream_next(void *stream, void **out_batch);
void lance_close_stream(void *stream);

void *lance_get_exec_schema(void *dataset, const uint8_t *exec_ir,
                            size_t exec_ir_len);
void *lance_create_dataset_exec_stream_ir(void *dataset, const uint8_t *exec_ir,
                                          size_t exec_ir_len);

int32_t lance_last_error_code();
const char *lance_last_error_message();
void lance_free_string(const char *s);

int64_t lance_dataset_count_rows(void *dataset);
int32_t lance_dataset_delete(void *dataset, const uint8_t *filter_ir,
                             size_t filter_ir_len, int64_t *out_deleted_rows);
int32_t lance_delete_transaction_with_storage_options(
    const char *path, const char **option_keys, const char **option_values,
    size_t options_len, const uint8_t *filter_ir, size_t filter_ir_len,
    void *session, void **out_transaction, int64_t *out_deleted_rows);

int32_t lance_dataset_add_columns(void *dataset,
                                  const ArrowSchema *new_columns_schema,
                                  const char **expressions,
                                  size_t expressions_len, uint32_t batch_size);
int32_t lance_dataset_drop_columns(void *dataset, const char **columns,
                                   size_t columns_len);
int32_t lance_dataset_alter_columns_rename(void *dataset, const char *path,
                                           const char *new_name);
int32_t lance_dataset_alter_columns_set_nullable(void *dataset,
                                                 const char *path,
                                                 uint8_t nullable);
int32_t lance_dataset_alter_columns_cast(void *dataset, const char *path,
                                         const ArrowSchema *new_type_schema);

int32_t lance_dataset_update_table_metadata(void *dataset, const char *key,
                                            const char *value);
int32_t lance_dataset_update_config(void *dataset, const char *key,
                                    const char *value);
int32_t lance_dataset_update_schema_metadata(void *dataset, const char *key,
                                             const char *value);
int32_t lance_dataset_update_field_metadata(void *dataset,
                                            const char *field_path,
                                            const char *key, const char *value);

int32_t lance_dataset_compact_files(void *dataset);
int32_t lance_dataset_compact_files_with_options(void *dataset,
                                                 const char *options_json,
                                                 const char **out_metrics_json);
int32_t lance_dataset_cleanup_old_versions(void *dataset,
                                           int64_t older_than_seconds,
                                           uint8_t delete_unverified);
int32_t lance_dataset_cleanup_old_versions_with_options(
    void *dataset, const char *options_json, const char **out_metrics_json);

const char *lance_dataset_list_config(void *dataset);
const char *lance_dataset_list_table_metadata(void *dataset);
const char *lance_dataset_list_schema_metadata(void *dataset);
const char *lance_dataset_list_field_metadata(void *dataset,
                                              const char *field_path);
const char *lance_dataset_list_indices(void *dataset);
int32_t lance_dataset_create_scalar_index(void *dataset, const char *column,
                                          const char *index_name,
                                          uint8_t replace);

uint64_t *lance_dataset_list_fragments(void *dataset, size_t *out_len);
void lance_free_fragment_list(uint64_t *ptr, size_t len);
typedef struct LanceFieldStats {
  uint32_t field_id;
  uint64_t bytes_on_disk;
} LanceFieldStats;

typedef struct LanceFragmentStats {
  uint64_t fragment_id;
  int64_t num_rows;
  uint64_t bytes_on_disk;
} LanceFragmentStats;

LanceFragmentStats *lance_dataset_list_fragment_stats(void *dataset,
                                                      size_t *out_len);
void lance_free_fragment_stats_list(LanceFragmentStats *ptr, size_t len);

LanceFieldStats *lance_dataset_list_field_stats(void *dataset, size_t *out_len);
void lance_free_field_stats_list(LanceFieldStats *ptr, size_t len);

typedef struct LanceNamedFieldStats {
  const char *name;
  uint64_t bytes_on_disk;
} LanceNamedFieldStats;

LanceNamedFieldStats *lance_dataset_list_named_field_stats(void *dataset,
                                                           size_t *out_len);
void lance_free_named_field_stats_list(LanceNamedFieldStats *ptr, size_t len);
void *lance_create_fragment_stream_ir(void *dataset, uint64_t fragment_id,
                                      const char **columns, size_t columns_len,
                                      const uint8_t *filter_ir,
                                      size_t filter_ir_len);
void *lance_create_dataset_stream_ir(void *dataset, const char **columns,
                                     size_t columns_len,
                                     const uint8_t *filter_ir,
                                     size_t filter_ir_len, int64_t limit,
                                     int64_t offset);
void *lance_create_dataset_sample_stream_ir(void *dataset, const char **columns,
                                            size_t columns_len,
                                            double sample_percentage,
                                            int64_t seed, uint8_t repeatable);
void *lance_create_dataset_take_stream(void *dataset, const uint64_t *row_ids,
                                       size_t row_ids_len, const char **columns,
                                       size_t columns_len);
void *lance_create_dataset_take_stream_unfiltered(void *dataset,
                                                  const uint64_t *row_ids,
                                                  size_t row_ids_len,
                                                  const char **columns,
                                                  size_t columns_len);

void *lance_open_writer_with_storage_options(
    const char *path, const char *mode, const char **option_keys,
    const char **option_values, size_t options_len, uint64_t max_rows_per_file,
    uint64_t max_rows_per_group, uint64_t max_bytes_per_file,
    const char *data_storage_version, void *session, const ArrowSchema *schema);
void *lance_open_uncommitted_writer_with_storage_options(
    const char *path, const char *mode, const char **option_keys,
    const char **option_values, size_t options_len, uint64_t max_rows_per_file,
    uint64_t max_rows_per_group, uint64_t max_bytes_per_file,
    const char *data_storage_version, void *session, const ArrowSchema *schema);
int32_t lance_writer_write_batch(void *writer, ArrowArray *array);
int32_t lance_writer_finish(void *writer);
int32_t lance_writer_finish_uncommitted(void *writer, void **out_transaction);
void lance_close_writer(void *writer);

int32_t lance_commit_transaction_with_storage_options(
    const char *path, const char **option_keys, const char **option_values,
    size_t options_len, void *session, void *transaction);
void lance_free_transaction(void *transaction);

int32_t lance_overwrite_update_transaction_with_irs_and_storage_options(
    const char *path, const char **option_keys, const char **option_values,
    size_t options_len, const uint8_t *predicate_ir, size_t predicate_ir_len,
    const char **set_columns, const uint8_t **set_expr_irs,
    const size_t *set_expr_ir_lens, size_t set_len, uint64_t max_rows_per_file,
    uint64_t max_rows_per_group, uint64_t max_bytes_per_file, void *session,
    void **out_transaction, uint64_t *out_rows_updated);

int32_t lance_merge_begin_with_storage_options(
    const char *path, const char **option_keys, const char **option_values,
    size_t options_len, uint64_t max_rows_per_file, uint64_t max_rows_per_group,
    uint64_t max_bytes_per_file, void *session, void **out_merge_handle);
int32_t lance_merge_add_delete_rowids(void *merge_handle,
                                      const uint64_t *row_ids,
                                      size_t row_ids_len);
int32_t lance_merge_add_insert_batch(void *merge_handle, void *array);
int32_t lance_merge_finish_uncommitted(void *merge_handle,
                                       void **out_transaction);
void lance_merge_abort(void *merge_handle);

const char *lance_explain_dataset_scan_ir(void *dataset, const char **columns,
                                          size_t columns_len,
                                          const uint8_t *filter_ir,
                                          size_t filter_ir_len, int64_t limit,
                                          int64_t offset, uint8_t verbose);

void *lance_get_knn_schema(void *dataset, const char *vector_column,
                           const float *query_values, size_t query_len,
                           uint64_t k, uint64_t nprobes, uint64_t refine_factor,
                           uint8_t prefilter, uint8_t use_index);
void *lance_create_knn_stream_ir(void *dataset, const char *vector_column,
                                 const float *query_values, size_t query_len,
                                 uint64_t k, uint64_t nprobes,
                                 uint64_t refine_factor,
                                 const uint8_t *filter_ir, size_t filter_ir_len,
                                 uint8_t prefilter, uint8_t use_index);

const char *lance_explain_knn_scan_ir(void *dataset, const char *vector_column,
                                      const float *query_values,
                                      size_t query_len, uint64_t k,
                                      uint64_t nprobes, uint64_t refine_factor,
                                      const uint8_t *filter_ir,
                                      size_t filter_ir_len, uint8_t prefilter,
                                      uint8_t use_index, uint8_t verbose);

void *lance_get_fts_schema(void *dataset, const char *text_column,
                           const char *query, uint64_t k, uint8_t prefilter);
void *lance_create_fts_stream_ir(void *dataset, const char *text_column,
                                 const char *query, uint64_t k,
                                 const uint8_t *filter_ir, size_t filter_ir_len,
                                 uint8_t prefilter);

void *lance_get_hybrid_schema(void *dataset);
void *lance_create_hybrid_stream_ir(void *dataset, const char *vector_column,
                                    const float *query_values, size_t query_len,
                                    const char *text_column,
                                    const char *text_query, uint64_t k,
                                    const uint8_t *filter_ir,
                                    size_t filter_ir_len, uint8_t prefilter,
                                    float alpha, uint32_t oversample_factor);

// Index DDL / metadata
int32_t lance_dataset_create_index(void *dataset, const char *index_name,
                                   const char **columns, size_t columns_len,
                                   const char *index_type,
                                   const char *params_json, uint8_t replace,
                                   uint8_t train);
int32_t lance_dataset_drop_index(void *dataset, const char *index_name);
int32_t lance_dataset_optimize_index(void *dataset, const char *index_name,
                                     uint8_t retrain);
int32_t
lance_dataset_optimize_index_with_options(void *dataset, const char *index_name,
                                          const char *options_json,
                                          const char **out_metrics_json);
void *lance_get_index_list_schema(void *dataset);
void *lance_create_index_list_stream(void *dataset);
char **lance_dataset_list_scalar_indexed_columns(void *dataset,
                                                 size_t *out_len);
void lance_free_scalar_indexed_columns(char **ptr, size_t len);

void lance_free_batch(void *batch);
int32_t lance_batch_to_arrow(void *batch, ArrowArray *out_array,
                             ArrowSchema *out_schema);
}
