# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## collision, eps = 16/255

| hash | n |  | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|
| pdq | 2 |  | 329 | 0.391 | 16.0 | 0.906 |
| phash | 2 |  | nan | 0.359 | 16.0 | 0.906 |

## evasion, eps = 4/255

| hash | n |  | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|
| pdq | 2 |  | nan | 0.07 | 4.0 | 0.985 |
| phash | 2 |  | nan | 0.078 | 3.0 | 0.992 |

## evasion, eps = 8/255

| hash | n |  | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|
| pdq | 2 |  | nan | 0.148 | 8.0 | 0.959 |
| phash | 2 |  | nan | 0.156 | 8.0 | 0.963 |
