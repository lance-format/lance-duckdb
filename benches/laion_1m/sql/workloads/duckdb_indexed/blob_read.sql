SELECT d.sample_id, d.caption, d.image_bytes
FROM blob_candidates c
JOIN laion_1m d USING (sample_id)
ORDER BY c.ord;
