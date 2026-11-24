"""More bootstrap comparisons from the saved predictions (no retraining).

Stacking vs each tree model, and text + source vs text only for every model.
Same bootstrap as train.py (1,000 resamples of story groups, seed 42).
Writes results/extra_comparisons.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
SEED, RESAMPLES = 42, 1000


def roc_auc(y, s):
    """ROC-AUC from ranks, so scikit-learn isn't needed."""
    ranks = pd.Series(s).rank().to_numpy()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


oof = pd.read_csv(RESULTS / "oof_predictions.csv")
scores = {key: part["score"].to_numpy() for key, part in oof.groupby(["features", "model"])}
first = oof.query("features == 'Text + Source' and model == 'Stacking'")
y, groups = first["label"].to_numpy(), first["group"].to_numpy()
for (features, model), part in oof.groupby(["features", "model"]):  # all models must have the same rows
    assert (part["label"].to_numpy() == y).all() and (part["group"].to_numpy() == groups).all()

by_group = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array).to_list()
rng = np.random.default_rng(SEED)
draws = [np.concatenate([by_group[g] for g in rng.integers(0, len(by_group), len(by_group))])
         for _ in range(RESAMPLES)]

MAIN = "Text + Source"
checks = [(f"Stacking vs {m}", (MAIN, "Stacking"), (MAIN, m))
          for m in ["Bagging (decision trees)", "Random Forest", "XGBoost"]]
checks += [("Adding the website to the text", (MAIN, m), ("Text only", m)) for m in oof["model"].unique()]

rows = []
for question, a, b in checks:
    diffs = [roc_auc(y[i], scores[a][i]) - roc_auc(y[i], scores[b][i]) for i in draws]
    low, high = np.percentile(diffs, [2.5, 97.5])
    rows.append({"question": question, "model": a[1], "compared with": f"{b[1]}, {b[0]}",
                 "difference": roc_auc(y, scores[a]) - roc_auc(y, scores[b]), "CI low": low, "CI high": high,
                 "real difference?": "yes" if low > 0 or high < 0 else "no (CI includes 0)"})
pd.DataFrame(rows).to_csv(RESULTS / "extra_comparisons.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
