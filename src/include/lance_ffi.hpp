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
void lance_close_dataset(void *dataset);

void *lance_get_schema(void *dataset);
void lance_free_schema(void *schema);
int32_t lance_schema_to_arrow(void *schema, ArrowSchema *out_schema);

void *lance_create_stream(void *dataset);
int32_t lance_stream_next(void *stream, void **out_batch);
void lance_close_stream(void *stream);

int32_t lance_last_error_code();
const char *lance_last_error_message();
void lance_free_string(const char *s);

int64_t lance_dataset_count_rows(void *dataset);

uint64_t *lance_dataset_list_fragments(void *dataset, size_t *out_len);
void lance_free_fragment_list(uint64_t *ptr, size_t len);
void *lance_create_fragment_stream(void *dataset, uint64_t fragment_id,
                                   const char **columns, size_t columns_len,
                                   const char *filter_sql);
void *lance_create_fragment_stream_ir(void *dataset, uint64_t fragment_id,
                                      const char **columns, size_t columns_len,
                                      const uint8_t *filter_ir,
                                      size_t filter_ir_len);

const char *lance_explain_dataset_scan_ir(void *dataset, const char **columns,
                                          size_t columns_len,
                                          const uint8_t *filter_ir,
                                          size_t filter_ir_len,
                                          uint8_t verbose);

void *lance_get_knn_schema(void *dataset, const char *vector_column,
                           const float *query_values, size_t query_len,
                           uint64_t k, uint8_t prefilter, uint8_t use_index);
void *lance_create_knn_stream(void *dataset, const char *vector_column,
                              const float *query_values, size_t query_len,
                              uint64_t k, const char *filter_sql,
                              uint8_t prefilter, uint8_t use_index);
void *lance_create_knn_stream_ir(void *dataset, const char *vector_column,
                                 const float *query_values, size_t query_len,
                                 uint64_t k, const uint8_t *filter_ir,
                                 size_t filter_ir_len, uint8_t prefilter,
                                 uint8_t use_index);

const char *lance_explain_knn_scan(void *dataset, const char *vector_column,
                                   const float *query_values, size_t query_len,
                                   uint64_t k, const char *filter_sql,
                                   uint8_t prefilter, uint8_t use_index,
                                   uint8_t verbose);
const char *lance_explain_knn_scan_ir(void *dataset, const char *vector_column,
                                      const float *query_values, size_t query_len,
                                      uint64_t k, const uint8_t *filter_ir,
                                      size_t filter_ir_len, uint8_t prefilter,
                                      uint8_t use_index, uint8_t verbose);

void *lance_get_fts_schema(void *dataset, const char *text_column,
                           const char *query, uint64_t k, uint8_t prefilter);
void *lance_create_fts_stream(void *dataset, const char *text_column,
                              const char *query, uint64_t k,
                              const char *filter_sql, uint8_t prefilter);
void *lance_create_fts_stream_ir(void *dataset, const char *text_column,
                                 const char *query, uint64_t k,
                                 const uint8_t *filter_ir, size_t filter_ir_len,
                                 uint8_t prefilter);

void *lance_get_hybrid_schema(void *dataset);
void *lance_create_hybrid_stream(void *dataset, const char *vector_column,
                                 const float *query_values, size_t query_len,
                                 const char *text_column,
                                 const char *text_query, uint64_t k,
                                 const char *filter_sql, uint8_t prefilter,
                                 float alpha, uint32_t oversample_factor);
void *lance_create_hybrid_stream_ir(void *dataset, const char *vector_column,
                                    const float *query_values, size_t query_len,
                                    const char *text_column,
                                    const char *text_query, uint64_t k,
                                    const uint8_t *filter_ir,
                                    size_t filter_ir_len, uint8_t prefilter,
                                    float alpha, uint32_t oversample_factor);

void lance_free_batch(void *batch);
int64_t lance_batch_num_rows(void *batch);
int32_t lance_batch_to_arrow(void *batch, ArrowArray *out_array,
                             ArrowSchema *out_schema);
}
