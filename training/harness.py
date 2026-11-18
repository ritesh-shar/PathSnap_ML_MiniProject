
"""
Member 3: PathSnap Model Training Harness

Models:
- Logistic Regression
- Random Forest
- LightGBM

Features:
- GroupKFold by trip_id
- Model comparison
- Test evaluation
- Save fitted LightGBM and Logistic Regression
- Save exact feature order and class labels
"""

from __future__ import annotations

import argparse
import time
import warnings
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42

DEFAULT_FEATURES = [
    "distance_m",
    "segment_length_m",
    "heading_difference_rad",
    "gps_speed_mps",
    "time_gap_s"
]


# ---------------------------------------------------------
# MODEL FACTORIES
# ---------------------------------------------------------

def _lr(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed)
    )


def _rf(seed):
    return RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=seed
    )


def _lgbm(seed):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )


def _knn(seed):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    )


def _mlp(seed):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
<<<<<<< feature/member3-models
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
=======
            hidden_layer_sizes=(32,),
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            batch_size=2048,
>>>>>>> main
            random_state=seed
        )
    )


def _gnb(seed):
    return GaussianNB()


MODEL_FACTORIES = {
    "logreg": _lr,
    "random_forest": _rf,
    "lightgbm": _lgbm,
    "knn": _knn,
    "mlp": _mlp,
    "gnb": _gnb
}

DEFAULT_MODELS = ["logreg", "random_forest", "lightgbm"]


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_table(path):

    path = Path(path)

    if path.is_dir():
        files = sorted(path.glob("*.csv"))

        if not files:
            raise SystemExit(f"No CSV files found in {path}")

        return pd.concat(
            (pd.read_csv(f) for f in files),
            ignore_index=True
        )

    if "*" in str(path):
        files = sorted(path.parent.glob(path.name))

        if not files:
            raise SystemExit(f"No files match {path}")

        return pd.concat(
            (pd.read_csv(f) for f in files),
            ignore_index=True
        )

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path)


def add_data_args(ap):

    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)

    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="trip_id")
    ap.add_argument("--point-col", default="observation_id")

    ap.add_argument(
        "--usable-col",
        default="label_usable"
    )

    ap.add_argument(
        "--feature-cols",
        default=",".join(DEFAULT_FEATURES)
    )

    ap.add_argument("--drop-cols", default="")


def split_xy(df, features, label_col, group_col, point_col):

    X = df[features].astype(float).to_numpy()
    y = df[label_col].astype(int).to_numpy()
    groups = df[group_col].to_numpy()

    keys = pd.DataFrame({
        "g": groups,
        "p": (
            df[point_col].to_numpy()
            if point_col in df.columns
            else np.arange(len(df))
        )
    })

    return X, y, groups, keys


def prepare_data(a):

    tr = load_table(a.train)
    te = load_table(a.test)

    if a.usable_col and a.usable_col in tr.columns:

        n0 = len(tr)
        m0 = len(te)

        tr = tr[tr[a.usable_col] == 1].copy()
        te = te[te[a.usable_col] == 1].copy()

        print(
            f"Usable rows: train {len(tr)}/{n0}, "
            f"test {len(te)}/{m0}"
        )

    if a.feature_cols == "auto":

        skip = {
            a.label_col,
            a.group_col,
            a.point_col,
            *[c for c in a.drop_cols.split(",") if c]
        }

        feats = [
            c for c in tr.select_dtypes(
                include=[np.number, "bool"]
            ).columns
            if c not in skip
        ]

    else:
        feats = [
            c for c in a.feature_cols.split(",") if c
        ]

    for c in feats:
        if c not in tr.columns or c not in te.columns:
            raise SystemExit(f"Missing feature: {c}")

    if a.label_col not in tr.columns or a.label_col not in te.columns:
        raise SystemExit("Label column missing")

    if a.group_col not in tr.columns:
        raise SystemExit("Group column missing from training data")

    if a.group_col not in te.columns:
        raise SystemExit("Group column missing from test data")

    print("Features:", feats)

    X, y, groups, keys = split_xy(
        tr, feats, a.label_col, a.group_col, a.point_col
    )

    Xt, yt, _, keys_t = split_xy(
        te, feats, a.label_col, a.group_col, a.point_col
    )

    return feats, X, y, groups, keys, Xt, yt, keys_t


# ---------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------

def top1_accuracy(keys, y, probabilities):

    d = keys.reset_index(drop=True).copy()

    d["y"] = np.asarray(y)
    d["score"] = np.asarray(probabilities)

    has_positive = (
        d.groupby(["g", "p"])["y"].transform("max") == 1
    )

    d = d[has_positive]

    if d.empty:
        return float("nan")

    top = (
        d.sort_values("score", ascending=False, kind="stable")
        .drop_duplicates(["g", "p"])
    )

    return float(top["y"].mean())


def score_proba(y, probabilities, keys):

    probabilities = np.clip(
        probabilities, 1e-15, 1 - 1e-15
    )

    return {
        "f1_macro": float(
            f1_score(
                y,
                (probabilities >= 0.5).astype(int),
                average="macro",
                zero_division=0
            )
        ),
        "logloss": float(
            log_loss(y, probabilities, labels=[0, 1])
        ),
        "top1": top1_accuracy(keys, y, probabilities)
    }


def _proba(model, X):

    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------
# GROUP CROSS VALIDATION
# ---------------------------------------------------------

def cross_validate_model(
    factory, seed, X, y, groups, keys, n_splits=5
):

    n_groups = len(np.unique(groups))

    if n_groups < 2:
        raise ValueError("At least two training trips are required")

    gkf = GroupKFold(
        n_splits=min(n_splits, n_groups)
    )

    rows = []

    for train_idx, val_idx in gkf.split(X, y, groups):

        model = factory(seed)

        start = time.perf_counter()

        model.fit(X[train_idx], y[train_idx])

        train_time = time.perf_counter() - start

        probabilities = _proba(model, X[val_idx])

        metrics = score_proba(
            y[val_idx],
            probabilities,
            keys.iloc[val_idx]
        )

        metrics["train_time_s"] = train_time

        rows.append(metrics)

    result = {
        k: float(np.nanmean([r[k] for r in rows]))
        for k in rows[0]
    }

    f1s = [r["f1_macro"] for r in rows]

    result["f1_macro_std"] = float(
        np.std(f1s, ddof=1)
    ) if len(f1s) > 1 else 0.0

    return result


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    name, factory, seed,
    X, y, groups, keys,
    Xt, yt, keys_t,
    n_splits=5
):

    cv = cross_validate_model(
        factory, seed, X, y, groups, keys, n_splits
    )

    model = factory(seed)

    start = time.perf_counter()

    model.fit(X, y)

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    probabilities = _proba(model, Xt)

    inference_time = time.perf_counter() - start

    test_metrics = score_proba(
        yt, probabilities, keys_t
    )

    return {
        "model": name,
        "cv_f1_macro": cv["f1_macro"],
        "cv_f1_macro_std": cv["f1_macro_std"],
        "cv_logloss": cv["logloss"],
        "cv_top1": cv["top1"],
        "cv_train_time_s_per_fold": cv["train_time_s"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_logloss": test_metrics["logloss"],
        "test_top1": test_metrics["top1"],
        "final_train_time_s": train_time,
        "test_inference_time_s": inference_time,
        "test_inference_ms_per_row": (
            1000 * inference_time / max(len(Xt), 1)
        ),
        "n_train_rows": len(X),
        "n_test_rows": len(Xt),
        "n_features": X.shape[1],
        "seed": seed
    }


# ---------------------------------------------------------
# SAVE FITTED MODELS FOR MEMBER 4
# ---------------------------------------------------------

def save_final_models(X, y, features, seed):

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train and save LightGBM
    print("\nTraining final LightGBM model...")

    lightgbm_model = _lgbm(seed)
    lightgbm_model.fit(X, y)

    joblib.dump(
        lightgbm_model,
        models_dir / "best_model.joblib"
    )

    print("Saved models/best_model.joblib")

    # Train and save Logistic Regression pipeline
    print("\nTraining final Logistic Regression model...")

    logistic_model = _lr(seed)
    logistic_model.fit(X, y)

    joblib.dump(
        logistic_model,
        models_dir / "logistic_model.joblib"
    )

    print("Saved models/logistic_model.joblib")

    # Save exact feature and class ordering
    metadata = {
        "feature_names": features,
        "classes": lightgbm_model.classes_.tolist(),
        "preprocessing": {
            "lightgbm": None,
            "logistic_regression": [
                "StandardScaler",
                "LogisticRegression"
            ]
        },
        "positive_class": 1,
        "negative_class": 0,
        "seed": seed
    }

    with open(
        models_dir / "feature_list.json",
        "w"
    ) as f:

        json.dump(metadata, f, indent=4)

    print("Saved models/feature_list.json")

    # Verify probability prediction
    print("\nVerifying saved models...")

    loaded_lgbm = joblib.load(
        models_dir / "best_model.joblib"
    )

    loaded_lr = joblib.load(
        models_dir / "logistic_model.joblib"
    )

    print(
        "LightGBM predict_proba shape:",
        loaded_lgbm.predict_proba(X[:5]).shape
    )

    print(
        "Logistic Regression predict_proba shape:",
        loaded_lr.predict_proba(X[:5]).shape
    )

    print("\nAll model files saved and verified successfully.")


# ---------------------------------------------------------
# OUTPUT FORMAT
# ---------------------------------------------------------

SPEC_COLUMNS = [
    "model",
    "cv_macro_f1",
    "cv_macro_f1_std",
    "test_top1_accuracy",
    "test_macro_f1",
    "test_log_loss",
    "inference_time_s",
    "train_time_s"
]


def to_spec_row(r):

    return {
        "model": r["model"],
        "cv_macro_f1": r["cv_f1_macro"],
        "cv_macro_f1_std": r["cv_f1_macro_std"],
        "test_top1_accuracy": r["test_top1"],
        "test_macro_f1": r["test_f1_macro"],
        "test_log_loss": r["test_logloss"],
        "inference_time_s": r["test_inference_time_s"],
        "train_time_s": r["final_train_time_s"]
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    add_data_args(ap)

    ap.add_argument(
        "--out",
        default="results/model_comparison.csv"
    )

    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS)
    )

    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)

    ap.add_argument(
        "--append",
        action="store_true"
    )

    a = ap.parse_args()

    np.random.seed(a.seed)

    (
        feats, X, y, groups, keys,
        Xt, yt, keys_t
    ) = prepare_data(a)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Training data must contain both binary classes 0 and 1"
        )

    results = []

    # Compare requested models
    for name in [
        m.strip() for m in a.models.split(",") if m.strip()
    ]:

        if name not in MODEL_FACTORIES:
            raise SystemExit(f"Unknown model: {name}")

        try:
            MODEL_FACTORIES[name](a.seed)

        except ImportError as e:
            print(f"Skipping {name}: {e}")
            continue

        print(f"\nEvaluating {name}...")

        result = evaluate_model(
            name,
            MODEL_FACTORIES[name],
            a.seed,
            X, y, groups, keys,
            Xt, yt, keys_t,
            a.n_splits
        )

        print(result)

        results.append(result)

    # Save fitted models for Member 4
    save_final_models(X, y, feats, a.seed)

    # Save comparison results
    out = Path(a.out)

    out.parent.mkdir(parents=True, exist_ok=True)

    detail_out = out.with_name(
        out.stem + "_detailed.csv"
    )

    outputs = [
        (
            out,
            pd.DataFrame(
                [to_spec_row(r) for r in results],
                columns=SPEC_COLUMNS
            )
        ),
        (
            detail_out,
            pd.DataFrame(results)
        )
    ]

    for path, df in outputs:

        if a.append and path.exists():
            df = pd.concat(
                [pd.read_csv(path), df],
                ignore_index=True
            )

        df.to_csv(path, index=False)

        print(f"Saved {path}")


if __name__ == "__main__":
    main()
