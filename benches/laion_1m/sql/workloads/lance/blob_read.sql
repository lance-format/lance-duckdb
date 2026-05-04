SELECT d.sample_id, d.caption, d.image_bytes
FROM blob_candidates c
JOIN 'benches/laion_1m/data/laion_1m_v22.lance' d USING (sample_id)
ORDER BY c.ord;
