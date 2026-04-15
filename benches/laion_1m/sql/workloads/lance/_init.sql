INSTALL lance;
LOAD lance;

CREATE OR REPLACE TEMP VIEW blob_candidates(sample_id, ord) AS
VALUES
  (647920324, 1),
  (2267729233, 2),
  (208262883, 3),
  (23143500, 4),
  (23159604, 5),
  (1689225656, 6),
  (12383, 7),
  (1388404165, 8),
  (1295846444, 9),
  (1666082858, 10),
  (393380069, 11),
  (694205194, 12),
  (46284090, 13),
  (1712362594, 14),
  (1318980498, 15),
  (2198303214, 16),
  (925607967, 17),
  (1990046737, 18),
  (254540961, 19),
  (1504103418, 20);

SET VARIABLE qvec = (
  SELECT img_emb::FLOAT[768]
  FROM read_parquet('benches/laion_1m/data/laion_1m_lz4.parquet')
  WHERE nsfw = 'UNLIKELY'
  LIMIT 1 OFFSET 12345
);
