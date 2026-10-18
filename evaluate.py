import os
import warnings

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ml_pipeline import bootstrap_ci, fmt, prepare_xy

warnings.filterwarnings("ignore")

# Everything lives in the data/ folder
TEST_SPLIT_PATH = os.path.join("data", "test_split.csv")
MODEL_PATH = "model.pkl"


def evaluate_saved_model(test_path=TEST_SPLIT_PATH, model_path=MODEL_PATH):
    """
    Loads the already-trained model.pkl (no training) and evaluates it
    directly on the held-out test split saved in data/test_split.csv.
    The test split keeps the natural prevalence (~9.5% depressed).
    """
    # ------------------------------------------------------------------
    # Load the existing trained model (pipeline + tuned threshold)
    # ------------------------------------------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model '{model_path}' not found. Run save_model.py first.")
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    print(f"Loaded model from: {model_path}")
    print(f"Model: {bundle['model_name']}   decision threshold: {threshold:.3f}")

    # ------------------------------------------------------------------
    # Load the saved test split
    # ------------------------------------------------------------------
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at '{test_path}'. Run save_model.py first.")

    print(f"Loading test split from: {test_path}")
    df = pd.read_csv(test_path)
    print("Detected CSV Columns:", df.columns.tolist())

    if "depression" not in df.columns:
        raise ValueError("Test split has no 'depression' target column.")

    X_test, y_test = prepare_xy(df)
    missing = [c for c in bundle["features"] if c not in X_test.columns]
    if missing:
        raise ValueError(f"Test split is missing feature columns: {missing}")
    X_test = X_test[bundle["features"]]
    print(f"Using 'depression' as the target column "
          f"({int(y_test.sum())} of {len(y_test)} depressed = {y_test.mean():.1%}).")

    # ------------------------------------------------------------------
    # Evaluate the pickled model on the held-out split
    # ------------------------------------------------------------------
    prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (prob >= threshold).astype(int)

    results = bootstrap_ci(y_test, prob, threshold)      # mean (95% bootstrap CI)

    print("\n" + "=" * 45)
    print(f"   ROC-AUC ON TEST SPLIT: {fmt(results['ROC_AUC'])}")
    print(f"   PR-AUC  ON TEST SPLIT: {fmt(results['PR_AUC'])}")
    print("=" * 45)

    print("\nAll metrics, mean (95% bootstrap CI):")
    for name, value in results.items():
        print(f"  {name:<13} {fmt(value)}")
    print(f"\n(A model with no skill would have PR-AUC of about {y_test.mean():.2f})")

    class_names = ["Not depressed", "Depressed"]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=class_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(pd.DataFrame(cm, index=[f"Actual: {c}" for c in class_names],
                       columns=[f"Predicted: {c}" for c in class_names]))


if __name__ == "__main__":
    evaluate_saved_model()