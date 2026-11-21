# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| certphash | 6 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.031 | 8.0 | 0.958 |
| dinohash | 6 | 16.7 | 66.7 | 66.7 | 83.3 | 1112 | 0.182 | 8.0 | 0.962 |
| dinohash_untrained | 6 | 33.3 | 50.0 | 100.0 | 100.0 | 1819 | 0.233 | 8.0 | 0.964 |
| neuralhash | 6 | 0.0 | 0.0 | 0.0 | 16.7 | 5960 | 0.177 | 8.0 | 0.97 |
