# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## collision, eps = 16/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash | 200 | 0.0 | 0.0 | 1.5 | 5.0 | 6725 | 0.361 | 15.92 | 0.926 |

## evasion, eps = 4/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash | 200 | 0.5 | 1.0 | 2.0 | 6.5 | 6356 | 0.172 | 3.99 | 0.992 |

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash | 200 | 5.5 | 11.0 | 27.0 | 41.5 | 3978 | 0.238 | 7.99 | 0.974 |
