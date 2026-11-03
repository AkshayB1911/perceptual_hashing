# Results summary: coco_run2

## Row counts

| Hash | Genuine rows | Failed attacks | Attacked images | Settings | Impostor pairs |
|---|---|---|---|---|---|
| dinohash_untrained | 88,000 | 0 | 1,000 | 88 | 200,000 |
| dinohash | 88,000 | 0 | 1,000 | 88 | 200,000 |

Images hashed: 5,000

## Headline numbers

**Mean match rate over all settings (%)**

| Hash | FPR 1e-2 | FPR 1e-3 | FPR 1e-4 | Published threshold | AUC |
|---|---|---|---|---|---|
| dinohash_untrained | 98.7 | **94.3** | 83.0 | – | 0.999 |
| dinohash | 98.0 | **89.7** | 75.4 | – | 0.999 |

**Match rate by attack family at FPR 1e-3**

| Hash | Compression | Filtering/noise | Photometric | Geometric | Flips | Overlays |
|---|---|---|---|---|---|---|
| dinohash_untrained | 0.94 | 0.93 | 0.99 | 0.95 | 0.64 | 0.89 |
| dinohash | 0.91 | 0.83 | 0.98 | 0.90 | 0.58 | 0.86 |

**95% bootstrap intervals by family at FPR 1e-3**

| Hash | Compression | Filtering/noise | Photometric | Geometric | Flips | Overlays |
|---|---|---|---|---|---|---|
| dinohash_untrained | 0.941-0.947 | 0.921-0.932 | 0.991-0.994 | 0.946-0.954 | 0.627-0.656 | 0.883-0.900 |
| dinohash | 0.905-0.912 | 0.824-0.838 | 0.977-0.982 | 0.897-0.909 | 0.570-0.595 | 0.847-0.866 |

## Thresholds and impostor distances

| Hash | Threshold, FPR 1e-3 | Published threshold | Impostor mean | Impostor std | Impostor min | Match rate below impostor min |
|---|---|---|---|---|---|---|
| dinohash_untrained | 0.2188 | – | 0.4811 | 0.0566 | 0.0938 | 58.8% |
| dinohash | 0.1771 | – | 0.4603 | 0.0571 | 0.0625 | 44.2% |

## Bit statistics

| Hash | Bits | Mean share of ones | Lowest bit | Highest bit | Sum of bit entropy | Mean abs. correlation |
|---|---|---|---|---|---|---|
| dinohash | 96 | 0.510 | 0.120 | 0.981 | 89.8 | 0.045 |
| dinohash_untrained | 96 | 0.486 | 0.001 | 0.727 | 93.0 | 0.041 |

## Settings compared with coco_run1_config.json

| Setting | Same? |
|---|---|
| attack_suite | yes |
| max_side | yes |
| max_pairs | yes |
| seed | yes |
| n_images | yes |
| n_attacked | yes |
