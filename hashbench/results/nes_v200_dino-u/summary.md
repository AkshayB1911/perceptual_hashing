# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## collision, eps = 16/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash_untrained | 200 | 0.0 | 0.0 | 0.0 | 1.5 | 8406 | 0.344 | 16.0 | 0.927 |

## evasion, eps = 4/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash_untrained | 200 | 9.5 | 16.5 | 27.0 | 47.5 | 4060 | 0.199 | 4.0 | 0.992 |

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| dinohash_untrained | 200 | 40.0 | 50.5 | 77.0 | 87.5 | 1477 | 0.226 | 7.98 | 0.971 |
