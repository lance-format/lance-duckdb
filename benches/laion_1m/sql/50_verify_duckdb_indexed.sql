LOAD fts;
LOAD vss;

SET scalar_subquery_error_on_multiple_rows=false;

CREATE OR REPLACE TEMP VIEW laion_1m_filtered AS
SELECT *
FROM laion_1m
WHERE nsfw = 'UNLIKELY'
  AND width >= 512;

SET VARIABLE qvec = (
  SELECT img_emb
  FROM laion_1m
  WHERE nsfw = 'UNLIKELY'
  LIMIT 1 OFFSET 12345
);

EXPLAIN ANALYZE
SELECT sample_id, caption, score
FROM (
  SELECT
    sample_id,
    caption,
    fts_main_laion_1m.match_bm25(sample_id, 'wedding bride', fields := 'caption') AS score
  FROM laion_1m_filtered
) t
WHERE score IS NOT NULL
ORDER BY score DESC, sample_id
LIMIT 20;

EXPLAIN ANALYZE
WITH needle(search_vec) AS (
  SELECT getvariable('qvec')
),
matches AS (
  SELECT
    m.row.sample_id AS sample_id,
    m.row.caption AS caption,
    m.score AS distance
  FROM needle,
       vss_match(laion_1m_filtered, search_vec, img_emb, 20) t,
       UNNEST(t.matches) AS u(m)
)
SELECT sample_id, caption, distance
FROM matches
ORDER BY distance ASC, sample_id
LIMIT 20;

EXPLAIN ANALYZE
SELECT sample_id, caption, array_distance(img_emb, getvariable('qvec')) AS distance
FROM laion_1m_filtered
ORDER BY distance ASC
LIMIT 20;

EXPLAIN ANALYZE
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
