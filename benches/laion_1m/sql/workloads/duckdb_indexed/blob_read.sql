SELECT d.sample_id, d.caption, octet_length(d.image_bytes) AS image_bytes_len
FROM blob_candidates c
JOIN laion_1m d USING (sample_id)
ORDER BY c.ord;
