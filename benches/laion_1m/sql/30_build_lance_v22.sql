INSTALL lance;
LOAD lance;

COPY (
  SELECT *
  FROM read_parquet('benches/laion_1m/data/laion_1m_lz4.parquet')
) TO 'benches/laion_1m/data/laion_1m_v22.lance'
  (FORMAT lance, mode 'overwrite', data_storage_version '2.2');

CREATE INDEX laion_1m_caption_inverted
ON 'benches/laion_1m/data/laion_1m_v22.lance' (caption)
USING INVERTED;

CREATE INDEX laion_1m_nsfw_btree
ON 'benches/laion_1m/data/laion_1m_v22.lance' (nsfw)
USING BTREE;

CREATE INDEX laion_1m_width_btree
ON 'benches/laion_1m/data/laion_1m_v22.lance' (width)
USING BTREE;

CREATE INDEX laion_1m_img_emb_ivf
ON 'benches/laion_1m/data/laion_1m_v22.lance' (img_emb)
USING IVF_FLAT WITH (num_partitions = 256, metric_type = 'l2');
