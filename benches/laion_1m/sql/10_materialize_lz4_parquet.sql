COPY (
  SELECT
    key::BIGINT AS sample_id,
    caption,
    NSFW AS nsfw,
    similarity,
    LICENSE AS license,
    url,
    width,
    height,
    original_width,
    original_height,
    md5,
    img_emb::FLOAT[768] AS img_emb,
    image.bytes AS image_bytes,
    image.path AS image_path
  FROM read_parquet('benches/laion_1m/data/source/default/partial-train/*.parquet', union_by_name = true)
  WHERE status = 'success'
    AND caption IS NOT NULL
    AND img_emb IS NOT NULL
    AND array_length(img_emb) = 768
    AND image.bytes IS NOT NULL
) TO 'benches/laion_1m/data/laion_1m_lz4.parquet'
  (FORMAT parquet, COMPRESSION lz4);
