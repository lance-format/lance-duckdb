SELECT sample_id, caption, score
FROM (
  SELECT
    sample_id,
    caption,
    fts_main_laion_1m.match_bm25(sample_id, 'wedding bride', fields := 'caption') AS score
  FROM laion_1m
  WHERE nsfw = 'UNLIKELY'
    AND width >= 512
) t
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT 20;
