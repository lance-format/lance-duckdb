SELECT p.sample_id, p.caption, p.image_bytes
FROM blob_candidates c
JOIN laion_1m_parquet p USING (sample_id)
ORDER BY c.ord;
