
import sys
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Exact feature order used throughout the project
FEATURE_COLUMNS = [
    "distance_m",
    "segment_length_m",
    "heading_difference_rad",
    "gps_speed_mps",
    "time_gap_s"
]

DATA_PATH = ROOT / "results" / "train_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"

# Optimized configuration from Member 3
BEST_PARAMS = {
    "n_estimators": 304,
    "learning_rate": 0.0305,
    "num_leaves": 65,
    "max_depth": 4,
    "min_child_samples": 69,
    "subsample": 0.687,
    "subsample_freq": 1,
    "colsample_bytree": 0.800,
    "reg_alpha": 0.0180,
    "reg_lambda": 2.15e-08,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1
}


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    df = df[df["label_usable"] == 1].copy()

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["label"].astype(int)

    print(f"Training rows: {len(X):,}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Class distribution:\n{y.value_counts()}")

    print("\nTraining optimized LightGBM...")
    model = LGBMClassifier(**BEST_PARAMS)
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nVerifying saved model...")
    loaded_model = joblib.load(MODEL_PATH)

    probabilities = loaded_model.predict_proba(X.iloc[:5])

    assert hasattr(loaded_model, "predict_proba")
    assert probabilities.shape == (5, 2)

    print("\nSUCCESS")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Classes: {loaded_model.classes_}")
    print(f"Probability shape: {probabilities.shape}")
    print(f"Feature order: {FEATURE_COLUMNS}")
    print(f"Model size: {MODEL_PATH.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
