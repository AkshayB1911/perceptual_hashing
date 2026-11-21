# NES black-box evasion, Sun et al. 2026 setting: summary

ASR (%) = verified evasions (plain hasher on the returned 8-bit image). Success: BER > 0.2 (>= 20 of 96 bits) for DinoHash / NeuralHash; CertPHash: its FPR 1e-3 L1 threshold. 'mean' = the average over the three budgets (Sun et al. Table 1).

| hash | l2 40 | l2 90 | l2 180 | mean |
|---|---|---|---|---|
| dinohash | 42.0 | 40.0 | 41.5 | 41.2 |
| dinohash_untrained | 58.0 | 53.0 | 53.0 | 54.7 |

| hash | l2 | n | ASR (95% CI) | attack-side ASR | agree % | median q | mean dist | l2 | l_inf ×255 | SSIM | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinohash | 40 | 200 | 42.0 (35.4-48.9) | 42.0 | 100.0 | 813.5 | 16.3 | 18.5 | 25.2 | 0.955 | 7.2 |
| dinohash | 90 | 200 | 40.0 (33.5-46.9) | 40.0 | 100.0 | 813.5 | 16.3 | 18.4 | 25.2 | 0.955 | 7.2 |
| dinohash | 180 | 200 | 41.5 (34.9-48.4) | 41.5 | 100.0 | 846 | 16.4 | 18.4 | 25.1 | 0.955 | 7.2 |
| dinohash_untrained | 40 | 200 | 58.0 (51.1-64.6) | 58.0 | 100.0 | 911 | 18.1 | 18.2 | 24.9 | 0.957 | 6.7 |
| dinohash_untrained | 90 | 200 | 53.0 (46.1-59.8) | 53.0 | 100.0 | 813.5 | 17.9 | 18.3 | 24.8 | 0.957 | 6.7 |
| dinohash_untrained | 180 | 200 | 53.0 (46.1-59.8) | 53.0 | 100.0 | 846 | 18.0 | 18.1 | 24.8 | 0.958 | 6.7 |
