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
