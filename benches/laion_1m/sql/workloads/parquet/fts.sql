WITH scored AS (
  SELECT
    sample_id,
    caption,
    CASE WHEN regexp_matches(lower(caption), 'wedding') THEN 1 ELSE 0 END
    + CASE WHEN regexp_matches(lower(caption), 'bride') THEN 1 ELSE 0 END AS lexical_score
  FROM laion_1m_filtered
)
SELECT sample_id, caption, lexical_score
FROM scored
WHERE lexical_score > 0
ORDER BY lexical_score DESC
LIMIT 20;
