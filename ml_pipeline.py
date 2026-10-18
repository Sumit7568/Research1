"""
Shared building blocks for the leakage-free depression-prediction pipeline.

Every data-dependent step (missing-column filter, imputation, scaling,
correlation filter, LASSO selection, undersampling) lives INSIDE one pipeline,
so it is fitted on training data only (or on the training part of each CV fold).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

SEED = 42
# coded variables that must be treated as categories, not as numbers
CAT_COLS = ["sex", "ethnicity", "education", "marital", "smoking", "drinking", "activity"]
NON_FEATURES = ["SEQN", "depression", "phq9_total"]   # label and PHQ-9 total are never features


def prepare_xy(df):
    """Split the merged table into X (features) and y (label)."""
    y = df["depression"].astype(int)
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns]).copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return X, y


# ----------------------------------------------------------------- custom steps
class DropHighMissing(BaseEstimator, TransformerMixin):
    """Drop columns with more than `max_missing` missing values (decided on training data)."""

    def __init__(self, max_missing=0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        self.keep_ = X.columns[X.isna().mean() <= self.max_missing].tolist()
        return self

    def transform(self, X):
        return X[self.keep_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove one variable of every pair with |correlation| above `threshold`."""

    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        n = corr.shape[0]
        drop = set()
        for j in range(n):
            if j in drop:
                continue
            for i in range(j):
                if i not in drop and corr[i, j] > self.threshold:
                    drop.add(j)
                    break
        self.keep_idx_ = np.array([k for k in range(n) if k not in drop])
        return self

    def transform(self, X):
        return X[:, self.keep_idx_]


def _cat_selector(X):
    return [c for c in X.columns if c in CAT_COLS]


def _num_selector(X):
    return [c for c in X.columns if c not in CAT_COLS]


def build_pipeline(model, cache_dir=None):
    """Full leakage-free pipeline ending in `model`."""
    numeric = SkPipeline([
        ("impute", IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=6, n_jobs=-1, random_state=SEED),
            max_iter=2, random_state=SEED)),
        ("scale", StandardScaler()),
    ])
    categorical = SkPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([("num", numeric, _num_selector), ("cat", categorical, _cat_selector)])
    lasso = SelectFromModel(
        LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=6, cv=3,
                             scoring="average_precision", class_weight="balanced",
                             random_state=SEED, max_iter=2000),
        threshold=1e-6)
    return Pipeline([
        ("missing", DropHighMissing(0.5)),
        ("prep", prep),
        ("corr", CorrelationFilter(0.8)),
        ("lasso", lasso),
        ("undersample", RandomUnderSampler(random_state=SEED)),   # active during fit only
        ("model", model),
    ], memory=cache_dir)


# ----------------------------------------------------------------- metrics
def compute_metrics(y_true, prob, threshold):
    y_true = np.asarray(y_true)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {
        "ROC_AUC": roc_auc_score(y_true, prob),
        "PR_AUC": average_precision_score(y_true, prob),
        "Accuracy": (tp + tn) / len(y_true),
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": prec,
        "F1": f1_score(y_true, pred, zero_division=0),
        "Balanced_acc": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
    }


def bootstrap_ci(y_true, prob, threshold, n_boot=1000, seed=SEED):
    """Point estimate and 95% bootstrap CI for every metric."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    rng = np.random.default_rng(seed)
    point = compute_metrics(y_true, prob, threshold)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() in (0, len(idx)):
            continue
        boots.append(compute_metrics(y_true[idx], prob[idx], threshold))
    boots = pd.DataFrame(boots)
    return {k: (point[k], boots[k].quantile(0.025), boots[k].quantile(0.975)) for k in point}


def fmt(ci_tuple):
    p, lo, hi = ci_tuple
    return f"{p:.2f} ({lo:.2f}-{hi:.2f})"