# DinoHash Table 1 reproduction (authors' APGD, bit accuracy %)

| Model | eps | n | Paper | Ours (float) | Ours (8-bit) | Median | mean linf (/255) |
|---|---|---|---|---|---|---|---|
| dinohash | 4/255 | 50 | 72.2 | 68.9 | 69.1 | 69.3 | 4.00 |
| dinohash | 8/255 | 50 | 64.8 | 56.1 | 56.2 | 55.7 | 8.00 |
| dinohash_untrained_paper | 4/255 | 50 | 0.3 | 0.0 | 0.0 | 0.0 | 4.00 |
| dinohash_untrained_paper | 8/255 | 50 | 0.0 | 0.0 | 0.0 | 0.0 | 8.00 |
