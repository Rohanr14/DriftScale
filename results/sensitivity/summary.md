## Final-stage BWT (signed; positive = no forgetting)

| Mapping | Source | VMs | Seeds | Naive BWT (95% CI) | Replay BWT (95% CI) | Reactive BWT (sanity) | Wilcoxon p (naive vs replay) | Naive mean-prior BWT | Replay mean-prior BWT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | azure | 32 | 5 | +5.25 [-221.86, +190.33] | +251.25 [+122.51, +426.04] | +0.00 [+0.00, +0.00] | 0.312 | -21.11 [-66.67, +29.55] | +90.08 [+58.54, +130.29] |
| convex | azure | 32 | 5 | -217.67 [-421.33, -68.24] | +206.67 [+89.98, +325.31] | +0.00 [+0.00, +0.00] | 0.062 | -88.59 [-121.05, -56.13] | +52.16 [+18.20, +86.13] |
| threshold | azure | 32 | 5 | -127.86 [-340.53, +73.27] | +134.86 [-188.47, +422.30] | +0.00 [+0.00, +0.00] | 0.125 | -38.74 [-96.91, +19.43] | +45.13 [-35.76, +121.29] |

## Per-checkpoint drift magnitude

| Checkpoint | Demand mean | Demand std | SMD vs ckpt 1 | KS vs ckpt 1 | Different? |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.907 | 0.518 | — | — | — |
| 25 | 1.829 | 0.149 | -2.828 | 0.986 | yes |
| 50 | 2.549 | 0.944 | -0.469 | 0.549 | yes |
| 75 | 1.570 | 0.309 | -3.134 | 0.943 | yes |
| 100 | 2.086 | 0.475 | -1.651 | 0.672 | yes |
| 125 | 2.663 | 0.303 | -0.574 | 0.370 | yes |
