# Adversarial sweep summary

Success rate (%) at the benign FPR threshold, with 95% Wilson interval. Evasion: the attacked copy is no longer matched. Collision: a false match is forced.

## Evasion

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| certphash | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| certphash_adv | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| certphash_base | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 40.0 (17–69) |
| certphash_ep2 | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| certphash_ep4 | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| certphash_ep8 | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| neuralhash | 10 | 0.0 (0–28) | 0.0 (0–28) | 10.0 (2–40) | 80.0 (49–94) |

Mean SSIM of the attacked images (evasion):

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| certphash | 0.997 | 0.990 | 0.964 | 0.895 |
| certphash_adv | 0.997 | 0.989 | 0.965 | 0.897 |
| certphash_base | 0.997 | 0.989 | 0.960 | 0.875 |
| certphash_ep2 | 0.997 | 0.991 | 0.968 | 0.902 |
| certphash_ep4 | 0.997 | 0.990 | 0.966 | 0.898 |
| certphash_ep8 | 0.997 | 0.990 | 0.969 | 0.907 |
| neuralhash | 0.995 | 0.982 | 0.940 | 0.835 |

## Collision

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| certphash | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 20.0 (6–51) |
| certphash_adv | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 30.0 (11–60) |
| certphash_base | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 40.0 (17–69) |
| certphash_ep2 | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| certphash_ep4 | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| certphash_ep8 | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) |
| neuralhash | 10 | 0.0 (0–28) | 0.0 (0–28) | 0.0 (0–28) | 30.0 (11–60) |

Mean SSIM of the attacked images (collision):

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| certphash | 0.997 | 0.990 | 0.966 | 0.900 |
| certphash_adv | 0.997 | 0.990 | 0.970 | 0.910 |
| certphash_base | 0.997 | 0.989 | 0.960 | 0.883 |
| certphash_ep2 | 0.997 | 0.990 | 0.967 | 0.899 |
| certphash_ep4 | 0.997 | 0.990 | 0.967 | 0.902 |
| certphash_ep8 | 0.997 | 0.991 | 0.970 | 0.911 |
| neuralhash | 0.995 | 0.982 | 0.939 | 0.836 |
