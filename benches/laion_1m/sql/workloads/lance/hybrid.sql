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
  prefilter = false
)
ORDER BY _hybrid_score DESC;
