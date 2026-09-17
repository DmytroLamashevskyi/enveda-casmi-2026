from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Mapping, Sequence

import numpy as np


def rank_from_scores(candidate_ids: Sequence[Hashable], scores: Sequence[float]):
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate_ids and scores must have the same length")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    return [candidate_ids[i] for i in order]


def reciprocal_rank_fusion(
    candidate_ids: Sequence[Hashable],
    score_views: Mapping[str, Sequence[float]],
    weights: Mapping[str, float] | None = None,
    k: float = 20.0,
) -> Dict[Hashable, float]:
    """Fuse heterogeneous score views by rank rather than raw score scale.

    Formula per view: weight / (k + rank), rank starts at 1.
    Returns candidate -> fused score.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    weights = dict(weights or {})
    fused = defaultdict(float)

    for view_name, scores in score_views.items():
        if len(scores) != len(candidate_ids):
            raise ValueError(
                f"View '{view_name}' has {len(scores)} scores for {len(candidate_ids)} candidates"
            )
        ranking = rank_from_scores(candidate_ids, scores)
        weight = float(weights.get(view_name, 1.0))
        for rank, candidate in enumerate(ranking, start=1):
            fused[candidate] += weight / (k + rank)

    return dict(fused)


def fused_ranking(
    candidate_ids: Sequence[Hashable],
    score_views: Mapping[str, Sequence[float]],
    weights: Mapping[str, float] | None = None,
    k: float = 20.0,
):
    scores = reciprocal_rank_fusion(candidate_ids, score_views, weights=weights, k=k)
    return sorted(scores, key=scores.get, reverse=True)


def topk_overlap(a: Sequence[Hashable], b: Sequence[Hashable], topk: int = 10) -> float:
    """Jaccard overlap of two Top-k lists. High overlap means an ensemble may add little diversity."""
    sa = set(a[:topk])
    sb = set(b[:topk])
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0
