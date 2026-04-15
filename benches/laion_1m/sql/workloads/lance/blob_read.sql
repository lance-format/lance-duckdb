SELECT d.sample_id, d.caption, octet_length(d.image_bytes) AS image_bytes_len
FROM blob_candidates c
JOIN 'benches/laion_1m/data/laion_1m_v22.lance' d USING (sample_id)
ORDER BY c.ord;
