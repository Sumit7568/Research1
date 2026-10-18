import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ml_pipeline import CAT_COLS, bootstrap_ci, fmt, prepare_xy

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_PATH = "model.pkl"


def get_full_data_path(data_path):
    """
    Always resolves the dataset inside the 'data/' directory.
    """
    return os.path.join(DATA_DIR, os.path.basename(data_path))


def calculate_phq9_score(responses):
    """
    Calculates total score and severity for PHQ-9 survey responses.
    Expects a list or iterable of 9 integers (0-3).
    (Rule-based screening only: it is NOT the machine learning model.)
    """
    total_score = sum(responses)

    # Determine severity based on standard 5-tier PHQ-9 scoring brackets
    if total_score <= 4:
        severity = "Minimal"
    elif total_score <= 9:
        severity = "Mild"
    elif total_score <= 14:
        severity = "Moderate"
    elif total_score <= 19:
        severity = "Moderately Severe"
    else:
        severity = "Severe"

    return total_score, severity


def load_model(model_path=MODEL_PATH):
    """Loads model.pkl: a dict with the pipeline, tuned threshold, model name and feature list."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model '{model_path}' not found. Run save_model.py first.")
    return joblib.load(model_path)


def predict_risk(values, model_path=MODEL_PATH, bundle=None):
    """
    Predicts depression risk (PHQ-9 >= 10) for ONE person. For use in app.py.

    values: dict of feature name -> value, for example
        {"age": 45, "sex": 2, "PIR": 1.2, "BMI": 31, "SBP": 128, "education": 3, ...}
    Missing features are allowed: the pipeline imputes them from the training data.
    Categorical codes follow NHANES: sex 1/2, ethnicity 1/2/3/4/6/7, education 1-5, marital 1-6.
    smoking: current/former/never, drinking: current/former/never, activity: mild/moderate/vigorous.

    Returns {"probability": float, "high_risk": bool, "threshold": float}
    """
    bundle = bundle or load_model(model_path)
    pipeline, threshold, features = bundle["pipeline"], bundle["threshold"], bundle["features"]

    # Match each categorical value to the category text the model saw during training
    # (for example 3 -> "3.0"), so the one-hot encoding lines up.
    ct = pipeline.named_steps["prep"]
    cat_cols = ct.transformers_[1][2]
    categories = dict(zip(cat_cols, ct.named_transformers_["cat"].named_steps["onehot"].categories_))

    row = {}
    for feat in features:
        v = values.get(feat, np.nan)
        if feat in CAT_COLS and not (isinstance(v, float) and np.isnan(v)) and v is not None:
            match = None
            for cat in categories.get(feat, []):
                try:
                    if float(cat) == float(v):
                        match = cat
                        break
                except (TypeError, ValueError):
                    if str(cat).lower() == str(v).strip().lower():
                        match = cat
                        break
            v = match if match is not None else np.nan
        row[feat] = np.nan if v is None else v

    X = pd.DataFrame([row], columns=features)
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(object)
    prob = float(pipeline.predict_proba(X)[:, 1][0])
    return {"probability": prob, "high_risk": prob >= threshold, "threshold": float(threshold)}


def calculate_model_accuracy(model_path=MODEL_PATH, data_path="test_split.csv"):
    """
    Evaluates the trained depression model against a dataset in the data/ folder
    (default test_split.csv, the held-out 20%) or a custom file name provided via CLI.

    Use test_split.csv for honest results. Evaluating on the full dataset includes the
    people the model was trained on, which makes the numbers look better than they are.
    """
    bundle = load_model(model_path)
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]

    full_data_path = get_full_data_path(data_path)
    if not os.path.exists(full_data_path):
        raise FileNotFoundError(f"Dataset '{full_data_path}' not found. Run save_model.py first.")

    print(f"Loading evaluation dataset from: {full_data_path}")
    if os.path.basename(full_data_path) != "test_split.csv":
        print("WARNING: this file is not the held-out test split. It may contain training "
              "rows, so the results can be over-optimistic.")

    df = pd.read_csv(full_data_path)
    if "depression" not in df.columns:
        raise ValueError("Dataset has no 'depression' target column.")

    X, y = prepare_xy(df)
    missing = [c for c in bundle["features"] if c not in X.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {missing}")
    X = X[bundle["features"]]          # same column order the model was trained with

    # Predict & Evaluate (tuned threshold, not the default 0.5)
    prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (prob >= threshold).astype(int)
    results = bootstrap_ci(y, prob, threshold)

    print(f"\nModel: {bundle['model_name']}   decision threshold: {threshold:.3f}")
    print("\n" + "=" * 45)
    print(f"   ROC-AUC: {fmt(results['ROC_AUC'])}")
    print(f"   PR-AUC : {fmt(results['PR_AUC'])}")
    print("=" * 45)

    print("\nAll metrics, mean (95% bootstrap CI):")
    for name, value in results.items():
        print(f"  {name:<13} {fmt(value)}")

    class_names = ["Not depressed", "Depressed"]
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred, labels=[0, 1], target_names=class_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print(pd.DataFrame(cm, index=[f"Actual: {c}" for c in class_names],
                       columns=[f"Predicted: {c}" for c in class_names]))


if __name__ == "__main__":
    # Accepts optional dataset file name from CLI; resolved inside data/
    custom_data_path = sys.argv[1] if len(sys.argv) > 1 else "test_split.csv"
    calculate_model_accuracy(data_path=custom_data_path)