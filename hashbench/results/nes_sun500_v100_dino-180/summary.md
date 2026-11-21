# NES black-box evasion, Sun et al. 2026 setting: summary

ASR (%) = verified evasions (plain hasher on the returned 8-bit image). Success: BER > 0.2 (>= 20 of 96 bits) for DinoHash / NeuralHash; CertPHash: its FPR 1e-3 L1 threshold. 'mean' = the average over the three budgets (Sun et al. Table 1).

| hash | l2 180 | mean |
|---|---|---|
| dinohash | 84.0 | 84.0 |

| hash | l2 | n | ASR (95% CI) | attack-side ASR | agree % | median q | mean dist | l2 | l_inf ×255 | SSIM | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinohash | 180 | 100 | 84.0 (75.6-89.9) | 84.0 | 100.0 | 1691 | 19.2 | 25.2 | 36.1 | 0.925 | 38.9 |
