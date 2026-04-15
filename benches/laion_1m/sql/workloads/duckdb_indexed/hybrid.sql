WITH vec_candidates AS (
  SELECT
    sample_id,
    array_distance(img_emb, getvariable('qvec')) AS distance
  FROM laion_1m_filtered
  ORDER BY distance ASC, sample_id
  LIMIT 200
),
vec AS (
  SELECT
    sample_id,
    row_number() OVER (ORDER BY distance ASC, sample_id) AS vector_rank
  FROM vec_candidates
),
fts AS (
  SELECT
    sample_id,
    row_number() OVER (ORDER BY score DESC, sample_id) AS lexical_rank
  FROM (
    SELECT
      sample_id,
      score
    FROM (
      SELECT
        sample_id,
        fts_main_laion_1m.match_bm25(sample_id, 'wedding bride', fields := 'caption') AS score
      FROM laion_1m_filtered
    ) t
    WHERE score IS NOT NULL
    ORDER BY score DESC, sample_id
    LIMIT 200
  ) ranked_fts
)
SELECT
  COALESCE(vec.sample_id, fts.sample_id) AS sample_id,
  COALESCE(0.5 / (60 + vec.vector_rank), 0.0) + COALESCE(0.5 / (60 + fts.lexical_rank), 0.0) AS hybrid_score
FROM vec
FULL OUTER JOIN fts USING (sample_id)
ORDER BY hybrid_score DESC, sample_id
LIMIT 20;
