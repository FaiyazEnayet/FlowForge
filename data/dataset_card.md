# Dataset card: book-state windows

- Sessions (chunks): 8
- Total seconds of book data: 14377 (3.99 hours)
- Symbols: BTCUSDT
- Manifest schema: 2
- Manifest SHA-256: eeca9db2c52c0a9d4ba6c05d79db85c226ec04a62e9522357433a26aa7e2c1d4
- Data transform version: 1
- Missingness: 3 of 14380 session-seconds have no usable bar (recording gaps; the bar after a gap is dropped)

## Split manifest (session-disjoint)

| Split | Sessions | Windows |
|-------|----------|---------|
| train | 5 | 828 |
| val | 1 | 168 |
| test | 2 | 336 |

## Sessions

| ID | Directory | Symbol | SHA256 | Split | Span (s) | Usable (s) | Missing (s) |
|----|-----------|--------|-------|-------|----------|------------|-------------|
| 0 | chunk1 | BTCUSDT | 5f423ce8ca226bd65c72d2bd8860ef2fc86c4c23c686072d60239d466394ec05 | train | 1797 | 1794 | 3 |
| 1 | chunk2 | BTCUSDT | 698e7088a69502a5741a7c66129dc21ea9ebfbb55fb38045651f3715da819d40 | train | 1798 | 1798 | 0 |
| 2 | chunk3 | BTCUSDT | 8826dbbb5c4b4a025bd38c32a35115cae6138c8660d4e6fef4427a25cc0c41d5 | train | 1798 | 1798 | 0 |
| 3 | chunk4 | BTCUSDT | ac95bd082f4c118194b78eb2509dbaa4f0f739b48d60c2e4c478370631a011d1 | train | 1797 | 1797 | 0 |
| 4 | chunk5 | BTCUSDT | 8c4c0867da217d264238de2ef1e5d075da768154fff1f90f53388715385dfa0e | train | 1796 | 1796 | 0 |
| 5 | chunk6 | BTCUSDT | 126a3e3ed74a8d4bc06ec8abd9ab3330bd9b111da5cfaddf8d97f314f7d8b313 | val | 1797 | 1797 | 0 |
| 6 | chunk7 | BTCUSDT | 62cfb8d708bfaf70be55ff11bb89ff60275a6aed0f3f63875affd7d626cec1d7 | test | 1798 | 1798 | 0 |
| 7 | chunk8 | BTCUSDT | 71455b02d6eb926b5f13f5611edfc2574c6fde65d278ba755cb2636005bd9339 | test | 1799 | 1799 | 0 |

## Notes
- Windows never cross chunk boundaries; every 60-row window spans exactly 59 seconds.
- No raw second appears in more than one split by construction (whole sessions assigned).
- Features standardized with train-split statistics only.