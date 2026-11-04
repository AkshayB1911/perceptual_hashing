# Adversarial sweep summary

Success rate (%) at the benign FPR threshold, with 95% Wilson interval. Evasion: the attacked copy is no longer matched. Collision: a false match is forced.

## Evasion

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| dinohash | 3 | 100.0 (44–100) | 100.0 (44–100) | 100.0 (44–100) | 100.0 (44–100) |
| dinohash_untrained | 3 | 100.0 (44–100) | 100.0 (44–100) | 100.0 (44–100) | 100.0 (44–100) |

Mean SSIM of the attacked images (evasion):

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| dinohash | 0.995 | 0.983 | 0.948 | 0.829 |
| dinohash_untrained | 0.994 | 0.984 | 0.948 | 0.842 |

## Collision

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| dinohash | 3 | 0.0 (0–56) | 0.0 (0–56) | 0.0 (0–56) | 0.0 (0–56) |
| dinohash_untrained | 3 | 100.0 (44–100) | 100.0 (44–100) | 100.0 (44–100) | 100.0 (44–100) |

Mean SSIM of the attacked images (collision):

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| dinohash | 0.995 | 0.985 | 0.941 | 0.835 |
| dinohash_untrained | 0.994 | 0.984 | 0.948 | 0.844 |
