# P1 counterfactual editing

Status: complete; 192/192 new edits and 96/96 references.

Values below are equal-weight case means, not independent-query estimates. Changed-fact success is undefined for rehearsal. D is undefined when no derived truth changes. S always contains 93 supervised facts.

| Width | Arm | Cases | Changed E | Heldout D | Conflict heldout D | Unseen U damage | Unchanged probe damage |
|---:|---|---:|---:|---:|---:|---:|---:|
| 256 | actual-only | 24 | 100.000% | — | — | 0.749% | 64.727% |
| 256 | class-balanced-exception | 24 | 100.000% | 46.007% | 23.873% | 0.486% | — |
| 256 | coherent | 24 | 100.000% | 82.726% | 84.977% | 0.464% | — |
| 256 | exception | 24 | 100.000% | 41.233% | 16.018% | 0.463% | — |
| 256 | old-fact-rehearsal | 24 | — | — | — | 0.081% | 0.000% |
| 256 | root-only | 24 | 100.000% | 0.260% | 0.000% | 0.156% | — |
| 768 | actual-only | 24 | 100.000% | — | — | 0.744% | 96.375% |
| 768 | class-balanced-exception | 24 | 100.000% | 46.788% | 15.221% | 0.326% | — |
| 768 | coherent | 24 | 100.000% | 95.660% | 97.921% | 0.332% | — |
| 768 | exception | 24 | 100.000% | 46.094% | 13.561% | 0.326% | — |
| 768 | old-fact-rehearsal | 24 | — | — | — | 0.022% | 0.000% |
| 768 | root-only | 24 | 100.000% | 0.087% | 0.174% | 0.067% | — |

## Balanced supervision versus uniform supervision

Positive D differences favor balancing; positive U differences mean more damage. Keep each model/chain paired. No E-matched postselection is used.

| Width | Paired cases | ΔE (pp) | Δconflict heldout D (pp) | Δunseen U (pp) |
|---:|---:|---:|---:|---:|
| 256 | 24 | +0.000 | +7.855 | +0.023 |
| 768 | 24 | +0.000 | +1.659 | -0.001 |

The paired CSV also contains root-only/actual-only response contrasts and the factorial interaction. These use fixed new-default/new-actual/old-default labels on the same probe IDs; own-world accuracy differences alone are not interpreted as routing effects. Rehearsal is active old-fact supervision, not an unchanged model. Step zero supplies the unchanged baseline.

`editing.csv` preserves all counts, known denominators, heldout versus supervised retention, cross-chain and independent-fact damage, and all planned checkpoints. `paired-editing.csv` retains width/world/initialization/organization/chain. No E/D/U composite or confirmatory significance is reported.
