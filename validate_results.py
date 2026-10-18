"""
validate_results.py  -  figures and stability check for the paper.

Part A (fast)  : ROC curve and precision-recall curve of the saved final model
                 on the held-out test split -> figures/roc_curve.png, figures/pr_curve.png
Part B (slower): repeats the experiment over several random train/test splits with the
                 SAME final model settings and reports mean +/- SD of every metric
                 -> results/stability_results.csv, figures/stability.png

Run:   python validate_results.py          (10 random splits)
       python validate_results.py 5        (5 random splits)
       python validate_results.py 0        (figures only, skip the stability check)

Needs: model.pkl, data/test_split.csv, data/nhanes_depression_dataset.csv, ml_pipeline.py
Install once:  python -m pip install matplotlib
"""
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from ml_pipeline import bootstrap_ci, compute_metrics, fmt, prepare_xy

warnings.filterwarnings("ignore")

MODEL_PATH = "model.pkl"
TEST_SPLIT_PATH = os.path.join("data", "test_split.csv")
FULL_DATA_PATH = os.path.join("data", "nhanes_depression_dataset.csv")
FIG_DIR, RES_DIR = "figures", "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def best_f1_threshold(y_true, prob):
    """Threshold that maximizes F1 (chosen on training predictions only)."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[np.argmax(f1)])


# ====================================================================== Part A: figures
bundle = joblib.load(MODEL_PATH)
pipeline, threshold, model_name = bundle["pipeline"], bundle["threshold"], bundle["model_name"]

X_test, y_test = prepare_xy(pd.read_csv(TEST_SPLIT_PATH))
X_test = X_test[bundle["features"]]
prob = pipeline.predict_proba(X_test)[:, 1]
prevalence = float(y_test.mean())

ci = bootstrap_ci(y_test, prob, threshold)
point = compute_metrics(y_test, prob, threshold)

# --- ROC curve
fpr, tpr, _ = roc_curve(y_test, prob)
fig, ax = plt.subplots(figsize=(5.5, 5.2))
ax.plot(fpr, tpr, lw=2, color="#2563eb", label=f"{model_name}, AUC {fmt(ci['ROC_AUC'])}")
ax.plot([0, 1], [0, 1], "--", color="gray", label="No skill (AUC 0.50)")
ax.scatter([1 - point["Specificity"]], [point["Sensitivity"]], color="#dc2626", zorder=5,
           label=f"Chosen threshold {threshold:.2f}")
ax.set_xlabel("1 - Specificity (false positive rate)")
ax.set_ylabel("Sensitivity (true positive rate)")
ax.set_title("ROC curve on the held-out test set")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=300)
plt.close(fig)

# --- Precision-recall curve
prec, rec, _ = precision_recall_curve(y_test, prob)
fig, ax = plt.subplots(figsize=(5.5, 5.2))
ax.plot(rec, prec, lw=2, color="#2563eb", label=f"{model_name}, PR-AUC {fmt(ci['PR_AUC'])}")
ax.axhline(prevalence, ls="--", color="gray", label=f"No skill (prevalence {prevalence:.3f})")
ax.scatter([point["Sensitivity"]], [point["Precision"]], color="#dc2626", zorder=5,
           label=f"Chosen threshold {threshold:.2f}")
ax.set_xlabel("Recall (sensitivity)")
ax.set_ylabel("Precision")
ax.set_ylim(0, 1.02)
ax.set_title("Precision-recall curve on the held-out test set")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "pr_curve.png"), dpi=300)
plt.close(fig)

print(f"Saved {FIG_DIR}/roc_curve.png and {FIG_DIR}/pr_curve.png "
      f"({model_name}, test n = {len(y_test)}, {int(y_test.sum())} depressed)")

# ====================================================================== Part B: stability
if N_SEEDS <= 0:
    sys.exit("Stability check skipped (0 seeds requested).")

print(f"\nStability check over {N_SEEDS} random splits (about 30-40 s per split)...")
X_all, y_all = prepare_xy(pd.read_csv(FULL_DATA_PATH))
X_all = X_all[bundle["features"]]

rows = []
for seed in range(N_SEEDS):
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2,
                                              random_state=seed, stratify=y_all)
    pipe = clone(pipeline).set_params(memory=None)      # same settings, refitted from scratch
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = best_f1_threshold(y_tr, oof)                  # threshold from training data only
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]

    m = compute_metrics(y_te, p, thr)
    m["Balanced_acc_at_0.5"] = compute_metrics(y_te, p, 0.5)["Balanced_acc"]
    m.update({"Seed": seed, "Threshold": thr, "Test_n": len(y_te), "Test_depressed": int(y_te.sum())})
    rows.append(m)
    print(f"  seed {seed}: ROC-AUC {m['ROC_AUC']:.3f}  PR-AUC {m['PR_AUC']:.3f}  "
          f"accuracy {m['Accuracy']:.3f}  sensitivity {m['Sensitivity']:.3f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES_DIR, "stability_results.csv"), index=False)

metrics = ["ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "Precision",
           "F1", "Balanced_acc", "Balanced_acc_at_0.5"]
summary = pd.DataFrame({
    "Mean": res[metrics].mean(),
    "SD": res[metrics].std(),
    "Min": res[metrics].min(),
    "Max": res[metrics].max(),
}).round(3)
summary.to_csv(os.path.join(RES_DIR, "stability_summary.csv"))
print(f"\nMean and SD over {N_SEEDS} random splits:")
print(summary.to_string())

# --- boxplot of the main metrics across splits
fig, ax = plt.subplots(figsize=(7, 4.5))
plot_metrics = ["ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "Balanced_acc"]
ax.boxplot([res[m] for m in plot_metrics], showmeans=True)
ax.set_xticklabels([m.replace("_", " ") for m in plot_metrics])
ax.set_ylabel("Value on test split")
ax.set_title(f"Stability across {N_SEEDS} random 80/20 splits")
ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "stability.png"), dpi=300)
plt.close(fig)
print(f"\nSaved {RES_DIR}/stability_results.csv, {RES_DIR}/stability_summary.csv and {FIG_DIR}/stability.png")