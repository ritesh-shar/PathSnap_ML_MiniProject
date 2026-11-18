
from pathlib import Path
import sys
import time

import joblib
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.harness import MODEL_FACTORIES

FEATURE_COLUMNS = [
    "distance_m",
    "segment_length_m",
    "heading_difference_rad",
    "gps_speed_mps",
    "time_gap_s",
]

TRAIN_FILE = ROOT / "results" / "train_features.csv"
MODEL_FILE = ROOT / "models" / "mlp.joblib"


def main():
    print("Loading training data...")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    df = pd.read_csv(TRAIN_FILE)

    # Keep only rows with usable labels.
    if "label_usable" in df.columns:
        df = df[df["label_usable"] == 1].copy()

    required = FEATURE_COLUMNS + ["label"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["label"].astype(int)

    print(f"Training rows: {len(df):,}")
    print(f"Features: {FEATURE_COLUMNS}")

    # Same MLP configuration used in the shared harness.
    model = MODEL_FACTORIES["mlp"](42)

    print("\nTraining MLP...")
    start = time.perf_counter()

    model.fit(X, y)

    elapsed = time.perf_counter() - start

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("\nTraining complete!")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Model saved to: {MODEL_FILE}")

    print("\nClasses:", model.classes_)


if __name__ == "__main__":
    main()
