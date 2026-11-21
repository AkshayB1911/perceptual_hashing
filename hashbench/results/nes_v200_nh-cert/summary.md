# NES black-box attack: summary

Success (%) within each query budget at the calibrated FPR 1e-3 threshold. Collision rows exclude pairs that already matched before the attack.

## collision, eps = 16/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| certphash | 200 | 3.0 | 3.5 | 5.5 | 6.5 | 1026 | 0.171 | 15.92 | 0.906 |
| certphash_adv | 200 | 3.5 | 4.5 | 7.5 | 13.0 | 4080.5 | 0.182 | 15.94 | 0.904 |
| certphash_base | 200 | 2.5 | 2.5 | 5.5 | 8.0 | 3732 | 0.178 | 15.87 | 0.906 |
| certphash_ep2 | 200 | 1.0 | 1.0 | 1.0 | 2.5 | 5249 | 0.168 | 15.96 | 0.911 |
| certphash_ep4 | 200 | 1.0 | 1.0 | 1.5 | 1.5 | 657 | 0.158 | 15.97 | 0.912 |
| certphash_ep8 | 198 | 0.0 | 0.0 | 0.0 | 0.5 | 6069 | 0.124 | 16.0 | 0.914 |
| neuralhash | 200 | 0.0 | 0.0 | 0.0 | 1.0 | 6684 | 0.409 | 15.44 | 0.948 |

## evasion, eps = 4/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| certphash | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.014 | 4.0 | 0.991 |
| certphash_adv | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.017 | 4.0 | 0.99 |
| certphash_base | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.018 | 4.0 | 0.991 |
| certphash_ep2 | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.016 | 4.0 | 0.992 |
| certphash_ep4 | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.013 | 4.0 | 0.992 |
| certphash_ep8 | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.008 | 4.0 | 0.993 |
| neuralhash | 200 | 0.5 | 0.5 | 0.5 | 0.5 | 247 | 0.091 | 3.94 | 0.994 |

## evasion, eps = 8/255

| hash | n | <=1k | <=2k | <=5k | <=10k | median queries | mean dist (norm) | l_inf ×255 | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| certphash | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.028 | 8.0 | 0.969 |
| certphash_adv | 200 | 0.0 | 0.0 | 0.5 | 0.5 | 4388 | 0.033 | 8.0 | 0.967 |
| certphash_base | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.036 | 8.0 | 0.967 |
| certphash_ep2 | 200 | 1.0 | 1.5 | 1.5 | 1.5 | 329 | 0.03 | 8.0 | 0.972 |
| certphash_ep4 | 200 | 0.0 | 0.5 | 0.5 | 1.0 | 4654.5 | 0.026 | 8.0 | 0.973 |
| certphash_ep8 | 200 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.016 | 8.0 | 0.974 |
| neuralhash | 200 | 1.0 | 1.0 | 3.0 | 5.0 | 3916.5 | 0.168 | 7.99 | 0.981 |
