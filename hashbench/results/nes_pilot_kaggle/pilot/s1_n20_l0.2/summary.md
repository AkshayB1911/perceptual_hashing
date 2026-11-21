# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| certphash | 6 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.03 | 8.0 | 0.964 |
| dinohash | 6 | 33.3 | 66.7 | 83.3 | 83.3 | 1108 | 0.186 | 8.0 | 0.964 |
| dinohash_untrained | 6 | 33.3 | 66.7 | 100.0 | 100.0 | 1415.5 | 0.229 | 8.0 | 0.965 |
| neuralhash | 6 | 0.0 | 0.0 | 0.0 | 16.7 | 7586 | 0.179 | 8.0 | 0.977 |
