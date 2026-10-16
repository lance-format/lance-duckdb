SELECT sample_id, caption, _distance
FROM lance_vector_search(
  'benches/laion_1m/data/laion_1m_v22.lance',
  'img_emb',
  getvariable('qvec'),
  k = 20,
  use_index = true,
  nprobs = 16,
  refine_factor = 2,
  prefilter = false
)
ORDER BY _distance ASC;
