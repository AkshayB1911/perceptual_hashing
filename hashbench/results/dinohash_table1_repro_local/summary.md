# DinoHash Table 1 reproduction (authors' APGD, bit accuracy %)

| Model | eps | n | Paper | Ours (float) | Ours (8-bit) | Median | mean linf (/255) |
|---|---|---|---|---|---|---|---|
| neuralhash | 4/255 | 50 | 0.1 | 75.6 | 75.6 | 75.5 | 4.00 |
| neuralhash | 8/255 | 50 | 0.0 | 58.4 | 58.4 | 57.3 | 8.00 |
| neuralhash_as_committed | 4/255 | 50 | 0.1 | 0.0 | 0.0 | 0.0 | 4.00 |
| neuralhash_as_committed | 8/255 | 50 | 0.0 | 0.0 | 0.0 | 0.0 | 8.00 |
