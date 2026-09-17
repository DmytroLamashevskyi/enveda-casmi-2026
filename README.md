# enveda-casmi-2026

Research/competition project for **Enveda CASMI 2026 — Molecule ID From Mass Spectra**.

Current Kaggle notebook: `enveda-casmi-2026-fast-spectral-cosine-baseline.ipynb`.

## V17 work

The next iteration focuses on making experiments trustworthy before adding more model complexity:

- molecule-level validation without spectrum leakage;
- pseudo Class-2 evaluation;
- Recall@25/50/100 and MRR@25;
- multi-view spectral preprocessing;
- Reciprocal Rank Fusion (RRF) for ensemble ranking.

See [`V17_GUIDE.md`](V17_GUIDE.md) for the step-by-step implementation guide.

Reusable helpers live in `src/validation.py` and `src/rank_fusion.py`.
