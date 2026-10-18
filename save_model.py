"""
save_model.py (updated for the research paper)

Trains six models on the NHANES 2013-2014 depression dataset with a leakage-free
pipeline, evaluates them on a held-out test set at natural prevalence, and saves
model.pkl (final model + tuned threshold) and data/test_split.csv.

Requires: ml_pipeline.py in the same folder and data/nhanes_depression_dataset.csv
Run:      python save_model.py
"""
import os
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

from ml_pipeline import SEED, bootstrap_ci, build_pipeline, fmt, prepare_xy

warnings.filterwarnings("ignore")

DATA_DIR = "data"
RESULTS_DIR = "results"


def get_full_data_path(data_path):
    """
    Always resolves the dataset inside the 'data/' directory.
    """
    return os.path.join(DATA_DIR, os.path.basename(data_path))


def get_models():
    """The six models of the base paper with small tuning grids."""
    return {
        "LR": (LogisticRegression(max_iter=2000, random_state=SEED),
               {"model__C": [0.01, 0.1, 1]}),
        "RF": (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
               {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [5, 10]}),
        "NB": (GaussianNB(),
               {"model__var_smoothing": [1e-9, 1e-7]}),
        "SVM": (SVC(probability=True, kernel="rbf", random_state=SEED),
                {"model__C": [0.1, 1, 10]}),
        "XGBoost": (XGBClassifier(subsample=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1),
                    {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.05, 0.1],
                     "model__n_estimators": [100, 200]}),
        "LightGBM": (LGBMClassifier(verbose=-1, random_state=SEED),
                     {"model__num_leaves": [4, 8], "model__learning_rate": [0.05, 0.1],
                      "model__n_estimators": [100, 200]}),
    }


def train_and_save_model(data_path="nhanes_depression_dataset.csv"):
    """
    Trains the depression-prediction models, exports model.pkl, and saves a separate
    held-out test split (data/test_split.csv) for evaluate.py.
    """
    full_data_path = get_full_data_path(data_path)
    if not os.path.exists(full_data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{full_data_path}'. Run prepare_data.py and put "
            f"nhanes_depression_dataset.csv in the '{DATA_DIR}/' folder.")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------- data + stratified split
    print(f"Loading dataset from: {full_data_path}")
    df = pd.read_csv(full_data_path)
    X, y = prepare_xy(df)          # PHQ-9 items/total are never features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"Train: {len(X_train)} ({y_train.sum()} depressed)   "
          f"Test: {len(X_test)} ({y_test.sum()} depressed)")

    # ---------------------------------------------------------- SAVE HELD-OUT TEST SPLIT (natural prevalence)
    test_split_path = os.path.join(DATA_DIR, "test_split.csv")
    pd.concat([X_test, y_test], axis=1).to_csv(test_split_path, index=False)
    print(f"Saved 20% held-out test split to: {test_split_path}")

    # ---------------------------------------------------------- train, tune and evaluate every model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cache_dir = tempfile.mkdtemp()      # caches imputation/LASSO so grid search stays fast
    rows, fitted = [], {}

    for name, (model, grid) in get_models().items():
        print(f"\n=== {name} ===")
        search = GridSearchCV(build_pipeline(model, cache_dir), grid,
                              scoring="average_precision", cv=cv, n_jobs=1, refit=True)
        search.fit(X_train, y_train)    # tuned on PR-AUC using training data only
        best = search.best_estimator_
        print(f"best params: {search.best_params_}   CV PR-AUC: {search.best_score_:.3f}")

        # decision threshold from out-of-fold TRAINING predictions (never from the test set)
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        prec, rec, thr = precision_recall_curve(y_train, oof)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
        threshold = float(thr[np.argmax(f1)])

        # single evaluation on the untouched, natural-prevalence test set
        prob = best.predict_proba(X_test)[:, 1]
        for label, t in [(name, threshold), (f"{name} (threshold 0.5)", 0.5)]:
            ci = bootstrap_ci(y_test, prob, t)
            rows.append({"Model": label, "CV_PR_AUC": round(search.best_score_, 3),
                         "Threshold": round(t, 3), **{k: fmt(v) for k, v in ci.items()}})
        fitted[name] = (best, threshold, search.best_score_)

    table2 = pd.DataFrame(rows)
    table2.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print("\n", table2.to_string(index=False))
    print(f"\nTest-set prevalence (PR-AUC chance level): {y_test.mean():.3f}")

    # ---------------------------------------------------------- final model = best CV score (not best test score)
    final_name = max(fitted, key=lambda n: fitted[n][2])
    final_model, final_thr, _ = fitted[final_name]
    joblib.dump({"pipeline": final_model, "threshold": final_thr,
                 "model_name": final_name, "features": X.columns.tolist()}, "model.pkl")
    print(f"\nModel successfully saved as 'model.pkl' -> {final_name}, threshold {final_thr:.3f}")

    # ---------------------------------------------------------- original-style protocol (Table 3)
    # Everything fitted on ALL data, then undersample ALL data, then split: the leaky approach.
    front = build_pipeline(XGBClassifier(), None)[:-2]    # missing filter, prep, corr filter, LASSO
    Xt = front.fit_transform(X, y)                        # fitted on full data (leakage)
    Xb, yb = RandomUnderSampler(random_state=SEED).fit_resample(Xt, y)
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, random_state=SEED, stratify=yb)
    leaky = XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    leaky_ci = bootstrap_ci(yte, leaky.predict_proba(Xte)[:, 1], 0.5)

    best_xgb, thr_xgb, _ = fitted["XGBoost"]
    p_xgb = best_xgb.predict_proba(X_test)[:, 1]
    table3 = pd.DataFrame([
        {"Protocol": "Original-style (leakage, balanced test set)", **{k: fmt(v) for k, v in leaky_ci.items()}},
        {"Protocol": "Corrected, threshold 0.5", **{k: fmt(v) for k, v in bootstrap_ci(y_test, p_xgb, 0.5).items()}},
        {"Protocol": "Corrected, tuned threshold", **{k: fmt(v) for k, v in bootstrap_ci(y_test, p_xgb, thr_xgb).items()}},
    ])
    table3.to_csv(os.path.join(RESULTS_DIR, "protocol_comparison.csv"), index=False)
    print("\n", table3.to_string(index=False))


if __name__ == "__main__":
    train_and_save_model()