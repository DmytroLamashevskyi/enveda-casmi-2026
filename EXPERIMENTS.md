# Experiment Log

Use this file as the source of truth for CASMI experiments. Every Kaggle submission should map to a Git commit and one row below.

## Result history

| ID | Version | Main idea | Local / diagnostic result | Public LB | Verified? | Notes |
|---|---|---|---|---:|---|---|
| E00 | V10–V11 | Library similarity + COCONUT precursor fallback | — | 0.163 | notebook-reported | Establishes candidate-database fallback |
| E01 | V14 | Confidence-threshold gating | — | 0.193 | notebook-reported | Gating between evidence regimes |
| E02 | V15 | Fragment + neutral-loss channels, multi-spectrum consensus | Class-2 ~0.170 in notebook notes | 0.205 | notebook-reported | First clear multi-channel gain |
| E03 | V16 | Mass-shifted analog propagation + MetFrag-lite + calibrated GBDT | Class-2 0.545 claimed in notebook validation | TBD | pending Kaggle submission | Do not record projected score as leaderboard result |

## Important diagnostic issue

The saved V16 run reports the following test-set behavior:

- `best_library_sim` mean = `1.0`
- `best_library_sim` std = `0.0`
- 100% of molecules classified by the dashboard as confident library matches

That is inconsistent with the notebook narrative that only a minority of test molecules should be direct Class-1 library matches. Before using library-confidence gating, verify whether:

- the direct-library calculation contains unintended self/duplicate matches;
- molecular keys collapse stereoisomers or otherwise merge structures;
- the diagnostic is measuring a candidate-level maximum that is not equivalent to an exact-structure library hit;
- train/test spectra contain duplicates not accounted for by the current Class-1 estimate.

Treat this as the highest-priority correctness check.

---

## E04 — Multi-configuration rank ensemble

### Hypothesis

Different tolerances and evidence weights will produce partially decorrelated ranking errors. Combining their ranks can outperform any single configuration.

### Initial members

| Member | PPM window | Peak tolerance | Analog power | Evidence |
|---|---:|---:|---:|---|
| A | 5 | 0.010 Da | 3 | current full pipeline |
| B | 10 | 0.005 Da | 3 | current full pipeline |
| C | 10 | 0.020 Da | 3 | current full pipeline |
| D | 10 | 0.010 Da | 2 | current full pipeline |
| E | 10 | 0.010 Da | 4 | current full pipeline |
| F | 20 | 0.010 Da | 3 | current full pipeline |

Do **not** ensemble all members automatically. First measure rank diversity.

### Diversity measurements

For every pair of members, record:

- Top-1 agreement
- Top-5 Jaccard similarity
- Spearman correlation of candidate ranks
- fraction of validation queries where only one member retrieves the true structure in Top-25

Highly correlated members add cost without useful diversity.

### Fusion methods to compare

#### 1. Weighted Reciprocal Rank Fusion

For candidate `c`:

```text
RRF(c) = Σ_m w_m / (k + rank_m(c))
```

Start with `k = 60` and equal weights, then tune only on held-out data.

#### 2. Weighted normalized-score fusion

Only use this after per-model score calibration. Raw probabilities/similarities from different runs are not necessarily comparable.

#### 3. Borda-style rank aggregation

Useful as a parameter-free sanity baseline.

### Success criterion

Promote E04 only if it improves held-out MRR@25 and does not rely solely on one public-leaderboard fluctuation.

---

## E05 — Evidence-channel ablation

Run the same validation split with:

1. direct library only
2. analog propagation only
3. MetFrag-lite only
4. library + analog
5. library + fragmentation
6. analog + fragmentation
7. all channels

Record both MRR@25 and candidate recall@25. This separates **retrieval failure** from **reranking failure**.

---

## E06 — Ranker validation and calibration

### Goals

- Build molecule-level train/validation folds.
- Prevent spectra from the same molecule/scaffold leaking across folds where possible.
- Train the GBDT only on training folds.
- Produce out-of-fold probabilities.
- Tune channel weights and thresholds on OOF predictions.

Metrics:

- MRR@25
- Recall@25
- Top-1 accuracy
- Top-5 accuracy
- calibration / confidence bins

---

## E07 — Conditional ensemble / confidence gating

Instead of one global blend, choose evidence weights from query confidence.

Possible features:

- best direct-library similarity
- margin between first and second library candidates
- best analog similarity
- number of candidates in the precursor-mass window
- fragmentation explain score
- number of spectra available for the molecule

Example strategy:

```text
very strong direct evidence -> trust library-heavy ranking
weak direct + strong analog  -> trust analog/fragment ensemble
weak evidence everywhere     -> use broad consensus / conservative rank fusion
```

This should only be attempted after the current `best_library_sim == 1.0` diagnostic is understood.

---

## Submission template

Copy this block for every submission:

```text
Experiment ID:
Date:
Git commit:
Kaggle notebook version:
Runtime:
Parameters changed:
Validation split / protocol:
Local MRR@25:
Recall@25:
Public LB:
Private LB:
What improved:
What regressed:
Next hypothesis:
```

## Rule

A leaderboard score without the exact code/configuration that produced it is not a reproducible experiment.
