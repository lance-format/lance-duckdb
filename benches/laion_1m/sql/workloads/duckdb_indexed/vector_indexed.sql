SELECT sample_id, caption, array_distance(img_emb, getvariable('qvec')) AS distance
FROM laion_1m_filtered
ORDER BY distance ASC, sample_id
LIMIT 20;
