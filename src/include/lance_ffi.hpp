#pragma once

#include "duckdb/common/arrow/arrow.hpp"

#include <cstddef>
#include <cstdint>

extern "C" {
void *lance_open_dataset(const char *path);
void *lance_open_dataset_with_storage_options(const char *path,
                                              const char **option_keys,
                                              const char **option_values,
                                              size_t options_len);
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
const char *lance_namespace_list_tables(const char *endpoint,
                                        const char *namespace_id,
                                        const char *bearer_token,
                                        const char *api_key,
                                        const char *delimiter);
void *lance_open_dataset_in_namespace(
    const char *endpoint, const char *table_id, const char *bearer_token,
    const char *api_key, const char *delimiter, const char **out_table_uri);
void lance_close_dataset(void *dataset);

void *lance_get_schema(void *dataset);
void lance_free_schema(void *schema);
int32_t lance_schema_to_arrow(void *schema, ArrowSchema *out_schema);

int32_t lance_stream_next(void *stream, void **out_batch);
void lance_close_stream(void *stream);

int32_t lance_last_error_code();
const char *lance_last_error_message();
void lance_free_string(const char *s);

int64_t lance_dataset_count_rows(void *dataset);
int32_t lance_dataset_delete(void *dataset, const uint8_t *filter_ir,
                             size_t filter_ir_len, int64_t *out_deleted_rows);

uint64_t *lance_dataset_list_fragments(void *dataset, size_t *out_len);
void lance_free_fragment_list(uint64_t *ptr, size_t len);
void *lance_create_fragment_stream_ir(void *dataset, uint64_t fragment_id,
                                      const char **columns, size_t columns_len,
                                      const uint8_t *filter_ir,
                                      size_t filter_ir_len);
void *lance_create_dataset_stream_ir(void *dataset, const char **columns,
                                     size_t columns_len,
                                     const uint8_t *filter_ir,
                                     size_t filter_ir_len, int64_t limit,
                                     int64_t offset);

void *lance_open_writer_with_storage_options(
    const char *path, const char *mode, const char **option_keys,
    const char **option_values, size_t options_len, uint64_t max_rows_per_file,
    uint64_t max_rows_per_group, uint64_t max_bytes_per_file,
    const ArrowSchema *schema);
void *lance_open_uncommitted_writer_with_storage_options(
    const char *path, const char *mode, const char **option_keys,
    const char **option_values, size_t options_len, uint64_t max_rows_per_file,
    uint64_t max_rows_per_group, uint64_t max_bytes_per_file,
    const ArrowSchema *schema);
int32_t lance_writer_write_batch(void *writer, ArrowArray *array);
int32_t lance_writer_finish(void *writer);
int32_t lance_writer_finish_uncommitted(void *writer, void **out_transaction);
void lance_close_writer(void *writer);

int32_t lance_commit_transaction_with_storage_options(
    const char *path, const char **option_keys, const char **option_values,
    size_t options_len, void *transaction);
void lance_free_transaction(void *transaction);

int32_t lance_overwrite_update_transaction_with_storage_options(
    const char *path, const char **option_keys, const char **option_values,
    size_t options_len, const char *predicate, const char **set_columns,
    const char **set_expressions, size_t set_len, uint64_t max_rows_per_file,
    uint64_t max_rows_per_group, uint64_t max_bytes_per_file,
    void **out_transaction, uint64_t *out_rows_updated);

const char *lance_explain_dataset_scan_ir(void *dataset, const char **columns,
                                          size_t columns_len,
                                          const uint8_t *filter_ir,
                                          size_t filter_ir_len, int64_t limit,
                                          int64_t offset, uint8_t verbose);

void *lance_get_knn_schema(void *dataset, const char *vector_column,
                           const float *query_values, size_t query_len,
                           uint64_t k, uint8_t prefilter, uint8_t use_index);
void *lance_create_knn_stream_ir(void *dataset, const char *vector_column,
                                 const float *query_values, size_t query_len,
                                 uint64_t k, const uint8_t *filter_ir,
                                 size_t filter_ir_len, uint8_t prefilter,
                                 uint8_t use_index);

const char *lance_explain_knn_scan_ir(void *dataset, const char *vector_column,
                                      const float *query_values,
                                      size_t query_len, uint64_t k,
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
void *lance_get_index_list_schema(void *dataset);
void *lance_create_index_list_stream(void *dataset);

void lance_free_batch(void *batch);
int32_t lance_batch_to_arrow(void *batch, ArrowArray *out_array,
                             ArrowSchema *out_schema);
}
