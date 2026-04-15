WITH needle(search_vec) AS (
  SELECT getvariable('qvec')
),
matches AS (
  SELECT
    m.row.sample_id AS sample_id,
    m.row.caption AS caption,
    m.score AS distance
  FROM needle,
       vss_match(laion_1m_filtered, search_vec, img_emb, 20) t,
       UNNEST(t.matches) AS u(m)
)
SELECT sample_id, caption, distance
FROM matches
ORDER BY distance ASC, sample_id;
