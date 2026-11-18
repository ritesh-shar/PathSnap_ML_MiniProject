import argparse, json, pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_table, score_proba


def load_model(path):
    try:
        import joblib
        m = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            m = pickle.load(f)
    if isinstance(m, dict):
        m = m.get("model", next(v for v in m.values() if hasattr(v, "predict_proba")))
    return m


def load_features(path):
    d = json.loads(Path(path).read_text())
    if isinstance(d, list):
        return d
    for k in ("features", "feature_columns", "feature_cols"):
        if k in d:
            return d[k]
    return next(v for v in d.values() if isinstance(v, list))


ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True)
ap.add_argument("--model", default="models/final_model.pkl")
ap.add_argument("--features", default="models/final_model_features.json")
ap.add_argument("--comparison", default="results/model_comparison.csv")
a = ap.parse_args()
out = Path("results")
out.mkdir(exist_ok=True)

model = load_model(a.model)
feats = load_features(a.features)
te = load_table(a.test)
if "label_usable" in te.columns:
    te = te[te["label_usable"] == 1]
proba = model.predict_proba(te[feats].astype(float).to_numpy())[:, 1]
keys = pd.DataFrame({"g": te["trip_id"].to_numpy(), "p": te["observation_id"].to_numpy()})
m = score_proba(te["label"].astype(int).to_numpy(), proba, keys)
print(f"Loaded model OK on {len(te)} test rows")
print({k: round(v, 4) for k, v in m.items()})

est = model.steps[-1][1] if hasattr(model, "steps") else model
if hasattr(est, "feature_importances_"):
    imp = pd.Series(est.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("Importance (split count)")
    ax.set_title("Final model feature importance")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    print("saved results/feature_importance.png")

if Path(a.comparison).exists():
    df = pd.read_csv(a.comparison).drop_duplicates("model", keep="last")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - 0.2, df["test_top1_accuracy"], 0.4, label="Test Top-1", color="#4C78A8")
    ax.bar(x + 0.2, df["test_macro_f1"], 0.4, label="Test macro F1", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on student_test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    print("saved results/model_comparison.png")