# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| certphash | 6 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.03 | 8.0 | 0.964 |
| dinohash | 6 | 16.7 | 16.7 | 33.3 | 33.3 | 1313 | 0.165 | 8.0 | 0.969 |
| dinohash_untrained | 6 | 0.0 | 0.0 | 0.0 | 66.7 | 7647.5 | 0.212 | 8.0 | 0.972 |
| neuralhash | 6 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.181 | 8.0 | 0.979 |
