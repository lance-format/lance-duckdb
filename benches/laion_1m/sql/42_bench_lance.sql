INSTALL lance;
LOAD lance;

SET VARIABLE qvec = (
  SELECT img_emb::FLOAT[768]
  FROM read_parquet('benches/laion_1m/data/laion_1m_lz4.parquet')
  WHERE nsfw = 'UNLIKELY'
  LIMIT 1 OFFSET 12345
);

.read benches/laion_1m/sql/workloads/lance/fts.sql
.read benches/laion_1m/sql/workloads/lance/vector_exact.sql
.read benches/laion_1m/sql/workloads/lance/vector_indexed.sql
.read benches/laion_1m/sql/workloads/lance/hybrid.sql
.read benches/laion_1m/sql/workloads/lance/blob_read.sql
