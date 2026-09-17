from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def group_holdout_split(
    df: pd.DataFrame,
    group_col: str,
    val_fraction: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by unique molecule/group id so the same molecule cannot leak across folds."""
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    groups = pd.Series(df[group_col].dropna().unique())
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(groups))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    val_groups = set(groups.iloc[perm[:n_val]].tolist())

    val_mask = df[group_col].isin(val_groups).to_numpy()
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    return train_idx, val_idx


def remove_validation_targets_from_library(
    library_df: pd.DataFrame,
    val_df: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    """Guard rail for pseudo-Class-2 evaluation: remove validation molecules from spectral library."""
    if key_col not in library_df.columns or key_col not in val_df.columns:
        raise KeyError(f"Missing key column: {key_col}")
    blocked = set(val_df[key_col].dropna().unique())
    return library_df.loc[~library_df[key_col].isin(blocked)].copy()


def assert_no_group_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_col: str,
) -> None:
    """Raise if any molecule/group occurs in both train and validation."""
    train_groups = set(train_df[group_col].dropna().unique())
    val_groups = set(val_df[group_col].dropna().unique())
    overlap = train_groups.intersection(val_groups)
    if overlap:
        sample = list(overlap)[:10]
        raise AssertionError(
            f"Group leakage detected: {len(overlap)} shared groups. Example: {sample}"
        )


def _truth_set(value) -> set:
    if isinstance(value, str):
        return {value}
    if isinstance(value, (set, list, tuple, np.ndarray, pd.Series)):
        return set(value)
    return {value}


def mrr_at_k(
    predictions: Mapping[Hashable, Sequence[Hashable]],
    truth: Mapping[Hashable, Hashable | Sequence[Hashable]],
    k: int = 25,
) -> float:
    """Mean Reciprocal Rank with zero credit below rank k."""
    rr: List[float] = []
    for qid, target in truth.items():
        candidates = predictions.get(qid, ())[:k]
        targets = _truth_set(target)
        score = 0.0
        for rank, candidate in enumerate(candidates, start=1):
            if candidate in targets:
                score = 1.0 / rank
                break
        rr.append(score)
    return float(np.mean(rr)) if rr else 0.0


def recall_at_k(
    predictions: Mapping[Hashable, Sequence[Hashable]],
    truth: Mapping[Hashable, Hashable | Sequence[Hashable]],
    k: int,
) -> float:
    """Fraction of queries whose true structure occurs anywhere in Top-k."""
    hits: List[float] = []
    for qid, target in truth.items():
        candidates = set(predictions.get(qid, ())[:k])
        targets = _truth_set(target)
        hits.append(float(bool(candidates.intersection(targets))))
    return float(np.mean(hits)) if hits else 0.0


def evaluate_ranking(
    predictions: Mapping[Hashable, Sequence[Hashable]],
    truth: Mapping[Hashable, Hashable | Sequence[Hashable]],
) -> Dict[str, float]:
    return {
        "recall@25": recall_at_k(predictions, truth, 25),
        "recall@50": recall_at_k(predictions, truth, 50),
        "recall@100": recall_at_k(predictions, truth, 100),
        "mrr@25": mrr_at_k(predictions, truth, 25),
    }
