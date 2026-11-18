
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathsnap.candidates import FEATURE_COLUMNS


def score_candidates(
    candidates: pd.DataFrame,
    model_path="models/best_model.joblib",
) -> pd.DataFrame:
    """Generate classifier probabilities for road candidates."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    # Only load a model supplied by a trusted teammate.
    model = joblib.load(model_path)

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in candidates.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = candidates[FEATURE_COLUMNS].astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature values contain NaN or infinity")

    if not hasattr(model, "predict_proba"):
        raise TypeError("Classifier must support predict_proba")

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Positive class 1 is missing")

    positive_index = classes.index(1)

    output = candidates.copy()
    output["candidate_probability"] = (
        model.predict_proba(X)[:, positive_index]
    )

    return output
