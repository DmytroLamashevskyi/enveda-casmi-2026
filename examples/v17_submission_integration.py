"""Minimal V17 submission integration example.

This file is intentionally small enough to copy into a Kaggle notebook cell.
It assumes your inference pipeline already produced a mapping:

    predictions[molecule_id] = [smiles_1, smiles_2, ...]

The sample submission defines the exact output contract and row order.
"""

from pathlib import Path

import pandas as pd

from src.submission import build_submission, validate_submission


# In the Kaggle notebook you already have SAMPLE from find_file('sample_submission.csv').
SAMPLE = Path("sample_submission.csv")

sample = pd.read_csv(SAMPLE)

# Replace this toy mapping with the result of your V17 fused ranking.
# Example:
# predictions = {
#     mid: fused_smiles_for_mid
#     for mid, fused_smiles_for_mid in inference_results.items()
# }
predictions = {
    mid: ["CCO"]
    for mid in sample["molecule_id"]
}

submission = build_submission(
    sample_submission=sample,
    predictions=predictions,
    topk=25,
    fallback="CCO",
)

validate_submission(submission, sample, topk=25)
submission.to_csv("submission.csv", index=False)

print(f"submission.csv ready: {submission.shape}")
print(submission.head())
