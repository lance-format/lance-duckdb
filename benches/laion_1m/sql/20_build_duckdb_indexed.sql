INSTALL fts;
LOAD fts;
INSTALL vss;
LOAD vss;

SET hnsw_enable_experimental_persistence = true;

DROP TABLE IF EXISTS laion_1m;

CREATE TABLE laion_1m AS
SELECT
  sample_id,
  caption,
  nsfw,
  similarity,
  license,
  url,
  width,
  height,
  original_width,
  original_height,
  md5,
  img_emb::FLOAT[768] AS img_emb,
  image_bytes,
  image_path
FROM read_parquet('benches/laion_1m/data/laion_1m_lz4.parquet');

CREATE INDEX laion_1m_sample_id_idx ON laion_1m(sample_id);
CREATE INDEX laion_1m_nsfw_idx ON laion_1m(nsfw);
CREATE INDEX laion_1m_width_idx ON laion_1m(width);
CREATE INDEX laion_1m_img_emb_hnsw_idx ON laion_1m USING HNSW (img_emb);

PRAGMA create_fts_index('laion_1m', 'sample_id', 'caption');
