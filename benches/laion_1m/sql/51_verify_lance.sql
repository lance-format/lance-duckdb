INSTALL lance;
LOAD lance;

SET VARIABLE qvec = (
  SELECT img_emb::FLOAT[768]
  FROM read_parquet('benches/laion_1m/data/laion_1m_lz4.parquet')
  WHERE nsfw = 'UNLIKELY'
  LIMIT 1 OFFSET 12345
);

EXPLAIN ANALYZE
SELECT sample_id, caption, _score
FROM lance_fts(
  'benches/laion_1m/data/laion_1m_v22.lance',
  'caption',
  'wedding bride',
  k = 20,
  prefilter = true
)
WHERE nsfw = 'UNLIKELY'
  AND width >= 512
ORDER BY _score DESC, sample_id;

EXPLAIN ANALYZE
SELECT sample_id, caption, _distance
FROM lance_vector_search(
  'benches/laion_1m/data/laion_1m_v22.lance',
  'img_emb',
  getvariable('qvec'),
  k = 20,
  use_index = false,
  prefilter = true
)
WHERE nsfw = 'UNLIKELY'
  AND width >= 512
ORDER BY _distance ASC, sample_id;

EXPLAIN ANALYZE
SELECT sample_id, caption, _distance
FROM lance_vector_search(
  'benches/laion_1m/data/laion_1m_v22.lance',
  'img_emb',
  getvariable('qvec'),
  k = 20,
  use_index = true,
  nprobs = 16,
  refine_factor = 2,
  prefilter = true
)
WHERE nsfw = 'UNLIKELY'
  AND width >= 512
ORDER BY _distance ASC, sample_id;

EXPLAIN ANALYZE
SELECT sample_id, caption, _hybrid_score, _distance, _score
FROM lance_hybrid_search(
  'benches/laion_1m/data/laion_1m_v22.lance',
  'img_emb',
  getvariable('qvec'),
  'caption',
  'wedding bride',
  k = 20,
  use_index = true,
  nprobs = 16,
  refine_factor = 2,
  alpha = 0.5,
  oversample_factor = 4,
  prefilter = true
)
WHERE nsfw = 'UNLIKELY'
  AND width >= 512
ORDER BY _hybrid_score DESC, sample_id;
