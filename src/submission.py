from __future__ import annotations

from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

import pandas as pd


def _clean_candidates(
    candidates: Sequence[str] | str | None,
    *,
    topk: int,
    fallback: str,
) -> list[str]:
    """Return exactly ``topk`` candidate strings while preserving ranking order.

    Duplicate structures are removed first because repeated candidates waste ranking
    slots. If fewer than ``topk`` unique structures are available, the list is padded
    with ``fallback`` (or the final candidate if fallback is empty).
    """
    if isinstance(candidates, str):
        raw = [x.strip() for x in candidates.split(";")]
    elif candidates is None:
        raw = []
    else:
        raw = [str(x).strip() for x in candidates]

    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) == topk:
            return out

    pad_value = fallback.strip() if fallback else (out[-1] if out else "CCO")
    while len(out) < topk:
        out.append(pad_value)
    return out


def build_submission(
    sample_submission: pd.DataFrame,
    predictions: Mapping[Hashable, Sequence[str] | str],
    *,
    id_col: str = "molecule_id",
    prediction_col: str = "smiles",
    topk: int = 25,
    fallback: str = "CCO",
    require_all_ids: bool = True,
) -> pd.DataFrame:
    """Build a competition submission using the sample file as the contract.

    Important: the sample file controls row order and expected IDs. We never assume a
    fixed test-set size, so the same code also works when Kaggle swaps in the hidden
    test set during submission inference.
    """
    required = {id_col, prediction_col}
    missing = required.difference(sample_submission.columns)
    if missing:
        raise KeyError(f"sample_submission is missing columns: {sorted(missing)}")

    if sample_submission[id_col].duplicated().any():
        raise ValueError(f"sample_submission contains duplicate {id_col} values")

    expected_ids = sample_submission[id_col].tolist()
    missing_ids = [qid for qid in expected_ids if qid not in predictions]
    if require_all_ids and missing_ids:
        preview = missing_ids[:10]
        raise KeyError(
            f"Missing predictions for {len(missing_ids)} IDs. Examples: {preview}"
        )

    rows: list[str] = []
    for qid in expected_ids:
        ranked = _clean_candidates(
            predictions.get(qid),
            topk=topk,
            fallback=fallback,
        )
        rows.append(";".join(ranked))

    out = sample_submission.copy()
    out[prediction_col] = rows
    validate_submission(
        out,
        sample_submission,
        id_col=id_col,
        prediction_col=prediction_col,
        topk=topk,
    )
    return out


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    id_col: str = "molecule_id",
    prediction_col: str = "smiles",
    topk: int = 25,
) -> None:
    """Fail fast when the output does not match the sample-submission contract."""
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError(
            "Submission columns/order differ from sample_submission: "
            f"expected {list(sample_submission.columns)}, got {list(submission.columns)}"
        )

    if len(submission) != len(sample_submission):
        raise ValueError(
            f"Submission row count differs from sample: {len(submission)} != {len(sample_submission)}"
        )

    if submission[id_col].tolist() != sample_submission[id_col].tolist():
        raise ValueError(
            f"{id_col} values/order must exactly match sample_submission"
        )

    counts = submission[prediction_col].fillna("").astype(str).map(
        lambda x: len(x.split(";")) if x else 0
    )
    bad = counts[counts != topk]
    if len(bad):
        raise ValueError(
            f"Every row must contain exactly {topk} semicolon-separated candidates; "
            f"{len(bad)} rows violate the contract"
        )

    empty = submission[prediction_col].fillna("").astype(str).map(
        lambda x: any(not item.strip() for item in x.split(";"))
    )
    if empty.any():
        raise ValueError("Submission contains empty candidate strings")


def write_submission(
    sample_path: str | Path,
    predictions: Mapping[Hashable, Sequence[str] | str],
    output_path: str | Path = "submission.csv",
    *,
    id_col: str = "molecule_id",
    prediction_col: str = "smiles",
    topk: int = 25,
    fallback: str = "CCO",
    require_all_ids: bool = True,
) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    submission = build_submission(
        sample,
        predictions,
        id_col=id_col,
        prediction_col=prediction_col,
        topk=topk,
        fallback=fallback,
        require_all_ids=require_all_ids,
    )
    submission.to_csv(output_path, index=False)
    return submission
