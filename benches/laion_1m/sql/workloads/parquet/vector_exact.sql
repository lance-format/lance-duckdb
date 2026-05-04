SELECT sample_id, caption, array_distance(img_emb::FLOAT[768], getvariable('qvec')) AS distance
FROM laion_1m_filtered
ORDER BY distance ASC
LIMIT 20;
