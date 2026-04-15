SELECT p.sample_id, p.caption, octet_length(p.image_bytes) AS image_bytes_len
FROM blob_candidates c
JOIN laion_1m_parquet p USING (sample_id)
ORDER BY c.ord;
