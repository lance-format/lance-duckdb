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
