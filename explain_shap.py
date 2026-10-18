"""
explain_shap.py  -  SHAP explainability for the saved final model.

Produces (like Figs. 1-4 of the base paper):
  figures/shap_bar.png          global importance (mean |SHAP|)
  figures/shap_beeswarm.png     direction and size of each feature's effect
  figures/shap_dependence_*.png value-vs-effect plots for the top features
  figures/shap_waterfall_example.png  explanation for one person
  results/shap_importance.csv   ranking table for the paper

Run:      python explain_shap.py
Needs:    model.pkl, data/test_split.csv, ml_pipeline.py
Install:  python -m pip install shap matplotlib
"""
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_pipeline import prepare_xy

warnings.filterwarnings("ignore")

MODEL_PATH = "model.pkl"
TEST_SPLIT_PATH = os.path.join("data", "test_split.csv")
FIG_DIR, RES_DIR = "figures", "results"
TOP_N = 6            # number of dependence plots
MAX_ROWS = 1000      # rows to explain (tree/linear); kernel models use fewer
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ------------------------------------------------------------ load model and data
bundle = joblib.load(MODEL_PATH)
pipe, features, model_name = bundle["pipeline"], bundle["features"], bundle["model_name"]
model = pipe.named_steps["model"]

X_test, y_test = prepare_xy(pd.read_csv(TEST_SPLIT_PATH))
X_test = X_test[features]

# ------------------------------------------------------------ transform with every step BEFORE the model
# (missing filter -> imputation/scaling/one-hot -> correlation filter -> LASSO).
# The undersampler is skipped: it only acts during fit.
front = pipe[:-2]
Xt = front.transform(X_test)

# feature names after each stage
names = pipe.named_steps["prep"].get_feature_names_out()
names = np.array([n.split("__", 1)[1] for n in names])          # drop "num__" / "cat__"
names = names[pipe.named_steps["corr"].keep_idx_]
names = names[pipe.named_steps["lasso"].get_support()]
assert len(names) == Xt.shape[1], "feature names do not match transformed matrix"

kernel_model = type(model).__name__ not in (
    "RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "LogisticRegression")
n_rows = 200 if kernel_model else MAX_ROWS
rng = np.random.default_rng(42)
idx = rng.choice(len(Xt), size=min(n_rows, len(Xt)), replace=False)
Xdf = pd.DataFrame(Xt[idx], columns=names)
raw_rows = X_test.iloc[idx]

# ------------------------------------------------------------ SHAP values
mname = type(model).__name__
if mname in ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier"):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xdf)
elif mname == "LogisticRegression":
    explainer = shap.LinearExplainer(model, Xdf)
    sv = explainer.shap_values(Xdf)
else:                                                            # SVM, Naive Bayes: slow but model-agnostic
    f = lambda d: model.predict_proba(d)[:, 1]
    explainer = shap.KernelExplainer(f, shap.kmeans(Xdf, 20))
    sv = explainer.shap_values(Xdf, nsamples=200)

base = explainer.expected_value
if isinstance(sv, list):             # older shap: one array per class
    sv = sv[1]
elif np.ndim(sv) == 3:               # newer shap: (rows, features, classes)
    sv = sv[:, :, 1]
base = float(np.atleast_1d(base)[-1])  # class-1 baseline

print(f"Explained {len(Xdf)} test participants with {model_name} ({len(names)} features).")

# ------------------------------------------------------------ importance table
imp = (pd.DataFrame({"Feature": names, "Mean_abs_SHAP": np.abs(sv).mean(axis=0)})
         .sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True))
imp.to_csv(os.path.join(RES_DIR, "shap_importance.csv"), index=False)
print(imp.head(10).to_string(index=False))

# ------------------------------------------------------------ Fig A: bar plot (global importance)
plt.figure(figsize=(7, 6))
shap.summary_plot(sv, Xdf, plot_type="bar", max_display=20, show=False)
plt.title(f"SHAP feature importance ({model_name})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=300)
plt.close()

# ------------------------------------------------------------ Fig B: beeswarm
plt.figure(figsize=(8, 6.5))
shap.summary_plot(sv, Xdf, max_display=20, show=False)
plt.title("SHAP summary (red = high value, blue = low value)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_beeswarm.png"), dpi=300)
plt.close()

# ------------------------------------------------------------ Fig C: dependence plots for the top features
for feat in imp["Feature"].head(TOP_N):
    plt.figure(figsize=(5.5, 4.5))
    shap.dependence_plot(feat, sv, Xdf, interaction_index=None, show=False)
    plt.title(f"SHAP dependence: {feat}")
    plt.tight_layout()
    safe = feat.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(FIG_DIR, f"shap_dependence_{safe}.png"), dpi=300)
    plt.close()

# ------------------------------------------------------------ Fig D: explanation for one person (highest predicted risk)
prob = model.predict_proba(Xdf)[:, 1]
i = int(np.argmax(prob))
expl = shap.Explanation(values=sv[i], base_values=base, data=Xdf.iloc[i].values,
                        feature_names=list(names))
plt.figure()
shap.plots.waterfall(expl, max_display=12, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_waterfall_example.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved plots to {FIG_DIR}/ and results/shap_importance.csv")
print("NOTE: numeric features are shown standardized (z-scores): 0 = average, +1 = one SD above.")