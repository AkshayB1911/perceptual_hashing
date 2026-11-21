# NES black-box evasion, Sun et al. 2026 setting: summary

ASR (%) = verified evasions (plain hasher on the returned 8-bit image). Success: BER > 0.2 (>= 20 of 96 bits) for DinoHash / NeuralHash; CertPHash: its FPR 1e-3 L1 threshold. 'mean' = the average over the three budgets (Sun et al. Table 1).

| hash | l2 40 | l2 90 | l2 180 | mean |
|---|---|---|---|---|
| dinohash | nan | nan | 84.0 | 84.0 |
| neuralhash | 63.0 | 69.0 | 63.0 | 65.0 |

| hash | l2 | n | ASR (95% CI) | attack-side ASR | agree % | median q | mean dist | l2 | l_inf ×255 | SSIM | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinohash | 180 | 100 | 84.0 (75.6-89.9) | 84.0 | 100.0 | 1691 | 19.2 | 25.2 | 36.1 | 0.925 | 38.9 |
| neuralhash | 40 | 100 | 63.0 (53.2-71.8) | 63.0 | 100.0 | 1821 | 17.1 | 24.0 | 33.9 | 0.926 | 31.4 |
| neuralhash | 90 | 100 | 69.0 (59.4-77.2) | 69.0 | 100.0 | 2406 | 17.6 | 24.6 | 34.7 | 0.924 | 28.6 |
| neuralhash | 180 | 100 | 63.0 (53.2-71.8) | 63.0 | 100.0 | 1886 | 17.2 | 24.2 | 34.1 | 0.926 | 30.7 |
