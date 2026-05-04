WITH vec_candidates AS (
  SELECT
    sample_id,
    array_distance(img_emb::FLOAT[768], getvariable('qvec')) AS distance
  FROM laion_1m_parquet
  ORDER BY distance ASC
  LIMIT 200
),
vec AS (
  SELECT
    sample_id,
    row_number() OVER (ORDER BY distance ASC) AS vector_rank
  FROM vec_candidates
),
fts_candidates AS (
  SELECT
    sample_id,
    CASE WHEN regexp_matches(lower(caption), 'wedding') THEN 1 ELSE 0 END
    + CASE WHEN regexp_matches(lower(caption), 'bride') THEN 1 ELSE 0 END AS lexical_score
  FROM laion_1m_parquet
),
fts AS (
  SELECT
    sample_id,
    row_number() OVER (ORDER BY lexical_score DESC) AS lexical_rank
  FROM (
    SELECT sample_id, lexical_score
    FROM fts_candidates
    WHERE lexical_score > 0
    ORDER BY lexical_score DESC
    LIMIT 200
  ) ranked_fts
)
SELECT
  COALESCE(vec.sample_id, fts.sample_id) AS sample_id,
  COALESCE(0.5 / (60 + vec.vector_rank), 0.0) + COALESCE(0.5 / (60 + fts.lexical_rank), 0.0) AS hybrid_score
FROM vec
FULL OUTER JOIN fts USING (sample_id)
ORDER BY hybrid_score DESC
LIMIT 20;
