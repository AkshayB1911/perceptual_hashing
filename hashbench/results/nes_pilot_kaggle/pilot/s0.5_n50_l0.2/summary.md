# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| certphash | 6 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.03 | 8.0 | 0.958 |
| dinohash | 6 | 33.3 | 50.0 | 66.7 | 83.3 | 1920 | 0.184 | 8.0 | 0.961 |
| dinohash_untrained | 6 | 16.7 | 83.3 | 83.3 | 83.3 | 1112 | 0.227 | 8.0 | 0.964 |
| neuralhash | 6 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.149 | 8.0 | 0.973 |
