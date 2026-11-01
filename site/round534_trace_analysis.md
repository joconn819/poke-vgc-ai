# Traced match loss analysis

- Input: `/home/justin/p/poke_learning_2/training/outputs/phase12_headless/round534_maxdamage_trace_round531_round532_32x2_20261031`
- Games: 256 across 32 team-pair clusters
- Checkpoint score: 43.75% (team-cluster 95% interval [30.08%, 57.42%])
- W/L/D: 112/144/0; action adjustments: 0

Action usage by outcome (fractions averaged per game; descriptive, not causal):

| Outcome | Games | Median decisions (p10–p90) | Move | Switch | Pass | Other | Top moves by mean per-game share |
|---|---:|---:|---:|---:|---:|---:|---|
| win | 112 | 8 (5–10) | 72.0% | 12.9% | 15.1% | 0.0% | rockslide (7.9%), heatwave (7.4%), closecombat (5.1%), moonblast (5.1%), hypervoice (5.0%), shadowball (4.4%) |
| loss | 144 | 7 (5–9) | 65.7% | 14.7% | 19.7% | 0.0% | heatwave (9.6%), shadowball (6.6%), hypervoice (4.4%), closecombat (4.1%), rockslide (3.8%), psychic (3.7%) |
| draw | 0 | 0 (0–0) | 0.0% | 0.0% | 0.0% | 0.0% |  |

Trace-length distribution (long matches can dominate totals):

| Outcome | Median trace steps (p10–p90) | Max | Matches over 200 steps |
|---|---:|---:|---:|
| win | 9 (6–12) | 13 | 0 |
| loss | 8 (6–10) | 13 | 0 |
| draw | 0 (0–0) | 0 | 0 |

Matched post-state trajectory by outcome:

| Outcome | Snapshot | Games | Policy alive | Opponent alive | Policy team HP sum | Opponent team HP sum |
|---|---:|---:|---:|---:|---:|---:|
| win | 1 | 112 | 5.71 | 5.57 | 5.10 | 4.77 |
| win | 2 | 112 | 5.15 | 4.60 | 4.65 | 4.18 |
| win | 3 | 112 | 4.74 | 4.07 | 4.12 | 3.65 |
| win | 4 | 110 | 4.24 | 3.56 | 3.62 | 3.12 |
| win | 5 | 98 | 3.84 | 3.19 | 3.18 | 2.69 |
| win | 6 | 85 | 3.59 | 2.64 | 2.92 | 2.34 |
| win | 7 | 44 | 3.45 | 2.61 | 2.90 | 2.25 |
| win | 8 | 18 | 3.56 | 2.67 | 2.87 | 2.25 |
| loss | 1 | 144 | 5.60 | 5.62 | 4.86 | 5.04 |
| loss | 2 | 144 | 4.72 | 5.19 | 4.33 | 4.63 |
| loss | 3 | 144 | 4.25 | 4.74 | 3.68 | 4.21 |
| loss | 4 | 138 | 3.65 | 4.38 | 3.13 | 3.76 |
| loss | 5 | 127 | 3.11 | 4.01 | 2.70 | 3.42 |
| loss | 6 | 102 | 2.51 | 3.75 | 2.23 | 3.13 |
| loss | 7 | 46 | 2.48 | 3.61 | 2.19 | 2.88 |
| loss | 8 | 20 | 2.60 | 3.60 | 2.18 | 2.79 |

**Limit:** Outcome-stratified actions and states are descriptive associations, not causal action values. The policy's side is recovered from the checkpoint_p1/p2 filename because the raw trace result is Player 1's result.
