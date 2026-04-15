SET scalar_subquery_error_on_multiple_rows=false;

SET VARIABLE qvec = (
  SELECT img_emb
  FROM laion_1m
  WHERE nsfw = 'UNLIKELY'
  LIMIT 1 OFFSET 12345
);

.read benches/laion_1m/sql/workloads/duckdb_indexed/fts.sql
.read benches/laion_1m/sql/workloads/duckdb_indexed/vector_exact.sql
.read benches/laion_1m/sql/workloads/duckdb_indexed/vector_indexed.sql
.read benches/laion_1m/sql/workloads/duckdb_indexed/hybrid.sql
.read benches/laion_1m/sql/workloads/duckdb_indexed/blob_read.sql
