# Adversarial sweep summary

Success rate (%) at the benign FPR threshold, with 95% Wilson interval. Evasion: the attacked copy is no longer matched. Collision: a false match is forced (random pairs, or the closest other image for `collision_closest`).

## Evasion

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| neuralhash | 200 | 0.5 (0–3) | 0.5 (0–3) | 18.0 (13–24) | 94.5 (90–97) |

Bit accuracy (%, bits still equal to the original; DinoHash paper metric):

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| neuralhash | 200 | 94.4 | 88.9 | 78.4 | 59.2 |

Mean SSIM of the attacked images:

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| neuralhash | 200 | 0.996 | 0.985 | 0.950 | 0.859 |

## Collision (random pairs)

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 | ε=16/255 | ε=32/255 |
|---|---|---|---|---|---|---|---|
| neuralhash | 200 | 0.0 (0–2) | 0.0 (0–2) | 3.0 (1–6) | 16.5 (12–22) | 56.5 (50–63) | 89.5 (84–93) |

Bit accuracy (%, bits equal to the target; DinoHash paper metric):

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 | ε=16/255 | ε=32/255 |
|---|---|---|---|---|---|---|---|
| neuralhash | 200 | 53.1 | 55.2 | 59.1 | 65.5 | 75.2 | 84.4 |

Mean SSIM of the attacked images:

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 | ε=16/255 | ε=32/255 |
|---|---|---|---|---|---|---|---|
| neuralhash | 200 | 0.996 | 0.986 | 0.957 | 0.876 | 0.735 | 0.580 |

## Collision (closest target)

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 | ε=16/255 | ε=32/255 |
|---|---|---|---|---|---|---|---|
| neuralhash | 200 | 38.0 (32–45) | 63.5 (57–70) | 80.5 (74–85) | 93.5 (89–96) | 97.0 (94–99) | 98.0 (95–99) |

Bit accuracy (%, bits equal to the target; DinoHash paper metric):

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 | ε=16/255 | ε=32/255 |
|---|---|---|---|---|---|---|---|
| neuralhash | 200 | 71.8 | 74.1 | 78.0 | 84.0 | 91.4 | 95.2 |

Mean SSIM of the attacked images:

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 | ε=16/255 | ε=32/255 |
|---|---|---|---|---|---|---|---|
| neuralhash | 200 | 0.996 | 0.986 | 0.955 | 0.882 | 0.763 | 0.684 |

## ε-monotonicity check

No drops: success and per-image progress never fell as ε grew.