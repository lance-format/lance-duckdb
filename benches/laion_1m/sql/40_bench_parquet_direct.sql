SET VARIABLE qvec = (
  SELECT img_emb::FLOAT[768]
  FROM read_parquet('benches/laion_1m/data/laion_1m_lz4.parquet')
  WHERE nsfw = 'UNLIKELY'
  LIMIT 1 OFFSET 12345
);

.read benches/laion_1m/sql/workloads/parquet/fts.sql
.read benches/laion_1m/sql/workloads/parquet/vector_exact.sql
.read benches/laion_1m/sql/workloads/parquet/vector_indexed.sql
.read benches/laion_1m/sql/workloads/parquet/hybrid.sql
.read benches/laion_1m/sql/workloads/parquet/blob_read.sql
