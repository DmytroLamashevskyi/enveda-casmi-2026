from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def group_holdout_split(
    df: pd.DataFrame,
    group_col: str,
    val_fraction: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by unique molecule/group id so the same structure cannot leak across folds."""
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    groups = pd.Series(df[group_col].dropna().unique())
    if len(groups) < 2:
        raise ValueError("Need at least two unique groups to build a holdout")

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(groups))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    n_val = min(n_val, len(groups) - 1)
    val_groups = set(groups.iloc[perm[:n_val]].tolist())

    val_mask = df[group_col].isin(val_groups).to_numpy()
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    return train_idx, val_idx


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


def remove_validation_targets_from_library(
    library_df: pd.DataFrame,
    val_df: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    """Remove validation structures from the spectral library for pseudo-Class-2 evaluation."""
    if key_col not in library_df.columns or key_col not in val_df.columns:
        raise KeyError(f"Missing key column: {key_col}")
    blocked = set(val_df[key_col].dropna().unique())
    return library_df.loc[~library_df[key_col].isin(blocked)].copy()


def make_pseudo_class2_split(
    train_df: pd.DataFrame,
    *,
    key_col: str = "inchikey14",
    val_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a leakage-safe pseudo-Class-2 benchmark from train.parquet.

    The validation structures are completely removed from the *spectral library*.
    Their structures may still exist in the candidate database, which mirrors CASMI
    Class 2: candidate structure known, reference spectrum unavailable.

    Returns:
        library_df: spectra allowed for retrieval/matching.
        val_df: held-out query spectra with known truth.
    """
    train_idx, val_idx = group_holdout_split(
        train_df,
        group_col=key_col,
        val_fraction=val_fraction,
        random_state=random_state,
    )
    library_df = train_df.iloc[train_idx].copy()
    val_df = train_df.iloc[val_idx].copy()
    assert_no_group_leakage(library_df, val_df, key_col)
    return library_df, val_df


def identity_truth(
    val_df: pd.DataFrame,
    *,
    key_col: str = "inchikey14",
) -> Dict[Hashable, Hashable]:
    """Build query->true-structure mapping when the query group key is the truth key.

    This is useful when local candidate rankings use the same structural key as the
    candidate pool (for the current notebook, `inchikey14`).
    """
    if key_col not in val_df.columns:
        raise KeyError(f"Missing key column: {key_col}")
    keys = val_df[key_col].dropna().drop_duplicates().tolist()
    return {key: key for key in keys}


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
