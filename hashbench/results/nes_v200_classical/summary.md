# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## collision, eps = 16/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| pdq | 200 | 75.0 | 88.5 | 96.0 | 98.5 | 616 | 0.389 | 15.85 | 0.941 |
| phash | 200 | 4.0 | 5.0 | 8.5 | 13.5 | 3650 | 0.348 | 15.62 | 0.95 |

## evasion, eps = 4/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| pdq | 200 | 0.0 | 0.0 | 0.0 | 0.5 | 5987 | 0.141 | 4.0 | 0.994 |
| phash | 200 | 0.0 | 0.0 | 0.5 | 1.0 | 5946 | 0.088 | 3.65 | 0.994 |

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| pdq | 200 | 0.0 | 0.5 | 2.5 | 5.0 | 4880 | 0.239 | 8.0 | 0.983 |
| phash | 200 | 2.0 | 2.0 | 4.5 | 6.5 | 3035 | 0.155 | 7.83 | 0.982 |
