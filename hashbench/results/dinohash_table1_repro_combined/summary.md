# DinoHash Table 1 reproduction, 50 V200 images, authors' APGD (100 steps), bit accuracy %

DinoHash rows: Kaggle GPU (results/dinohash_table1_repro_kaggle). NeuralHash rows: local CPU (results/dinohash_table1_repro_local).

| Model | eps | Paper | Ours | Ours (8-bit) |
|---|---|---|---|---|
| dinohash | 4/255 | 72.2 | 68.9 | 69.1 |
| dinohash | 8/255 | 64.8 | 56.1 | 56.2 |
| dinohash_untrained_paper | 4/255 | 0.3 | 0.0 | 0.0 |
| dinohash_untrained_paper | 8/255 | 0.0 | 0.0 | 0.0 |
| neuralhash | 4/255 | 0.1 | 75.6 | 75.6 |
| neuralhash | 8/255 | 0.0 | 58.4 | 58.4 |
| neuralhash_as_committed | 4/255 | 0.1 | 0.0 | 0.0 |
| neuralhash_as_committed | 8/255 | 0.0 | 0.0 | 0.0 |
