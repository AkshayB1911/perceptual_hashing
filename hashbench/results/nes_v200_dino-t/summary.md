# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## collision, eps = 16/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.361 | 15.92 | 0.926 |

## evasion, eps = 4/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash | 200 | 8.0 | 13.0 | 27.0 | 39.5 | 3240 | 0.154 | 3.98 | 0.992 |

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash | 200 | 43.5 | 50.0 | 71.0 | 82.0 | 923.5 | 0.184 | 7.94 | 0.971 |
