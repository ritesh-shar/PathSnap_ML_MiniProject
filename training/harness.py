"""Member 3: training harness for PathSnap candidate-arc classification.

Input: features CSV/parquet produced by Member 2's build_candidates, one row per
(GPS point, candidate arc). Needs a trip id column, a point id column, a binary
label column (1 = candidate is the true arc) and numeric feature columns.

Usage:
    python training/harness.py --train features_train.csv --test features_test.csv

Defaults (override with flags): label=label, group=trip_id, point=point_idx.
Everything else numeric is a feature unless listed in --drop-cols.

Metrics per model (CV = GroupKFold by trip_id on train only; test = student_test):
    macro F1 (threshold 0.5), log loss, per-point Top-1 accuracy, train time,
    inference time. Seeds fixed.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

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


# ----------------------------------------------------------------------------
# Model registry. Each entry is a function seed -> fresh, unfitted estimator.
# Member 4 can add to / use this registry (knn, mlp, gnb are already here).
# ----------------------------------------------------------------------------
def _lr(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))


def _rf(seed):
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)


def _lgbm(seed):
    from lightgbm import LGBMClassifier  # ImportError if not installed

    return LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed,
                          n_jobs=-1, verbose=-1)


def _knn(seed):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=-1))


def _mlp(seed):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            batch_size=2048,
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
    "gnb": _gnb,
}
DEFAULT_MODELS = ["logreg", "random_forest", "lightgbm"]


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
DEFAULT_FEATURES = ["distance_m", "segment_length_m", "heading_difference_rad",
                    "gps_speed_mps", "time_gap_s"]  # = FEATURE_COLUMNS in pathsnap/candidates.py


def load_table(path: str | Path) -> pd.DataFrame:
    """One CSV/parquet file, a folder of per-trip CSVs (results/features/train), or a glob."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise SystemExit(f"no .csv files in {path}")
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    if "*" in str(path):
        files = sorted(path.parent.glob(path.name))
        if not files:
            raise SystemExit(f"no files match {path}")
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def add_data_args(ap: argparse.ArgumentParser):
    ap.add_argument("--train", required=True, help="file, folder of per-trip CSVs, or glob")
    ap.add_argument("--test", required=True, help="file, folder of per-trip CSVs, or glob")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="trip_id")
    ap.add_argument("--point-col", default="observation_id")
    ap.add_argument("--usable-col", default="label_usable",
                    help="keep only rows where this is 1 (empty string disables)")
    ap.add_argument("--feature-cols", default=",".join(DEFAULT_FEATURES),
                    help="comma-separated feature columns, or 'auto' for every numeric column "
                         "except label/ids (risky: can leak reference_* columns)")
    ap.add_argument("--drop-cols", default="", help="with --feature-cols auto: extra columns to exclude")


def prepare_data(a):
    """Load train/test, filter unusable labels, return features and arrays."""
    tr, te = load_table(a.train), load_table(a.test)
    if a.usable_col and a.usable_col in tr.columns:
        n0, m0 = len(tr), len(te)
        tr, te = tr[tr[a.usable_col] == 1], te[te[a.usable_col] == 1]
        print(f"kept usable-label rows: train {len(tr)}/{n0}, test {len(te)}/{m0}")
    if a.feature_cols == "auto":
        feats = get_feature_columns(tr, a.label_col, a.group_col, a.point_col,
                                    [c for c in a.drop_cols.split(",") if c])
    else:
        feats = [c for c in a.feature_cols.split(",") if c]
    for c in feats:
        if c not in tr.columns or c not in te.columns:
            raise SystemExit(f"feature column '{c}' missing from train or test data")
    if a.point_col not in tr.columns:
        warnings.warn(f"point column '{a.point_col}' not found; Top-1 will be meaningless")
    print(f"{len(feats)} features: {feats}")
    X, y, groups, keys = split_xy(tr, feats, a.label_col, a.group_col, a.point_col)
    Xt, yt, _, keys_t = split_xy(te, feats, a.label_col, a.group_col, a.point_col)
    return feats, X, y, groups, keys, Xt, yt, keys_t


def get_feature_columns(df: pd.DataFrame, label_col: str, group_col: str,
                        point_col: str, drop_cols=()) -> list[str]:
    skip = {label_col, group_col, point_col, *drop_cols}
    return [c for c in df.select_dtypes(include=[np.number, "bool"]).columns if c not in skip]


def split_xy(df: pd.DataFrame, features: list[str], label_col: str, group_col: str, point_col: str):
    X = df[features].astype(float).to_numpy()
    y = df[label_col].astype(int).to_numpy()
    groups = df[group_col].to_numpy()
    keys = pd.DataFrame({"g": groups,
                         "p": df[point_col].to_numpy() if point_col in df.columns else np.arange(len(df))})
    return X, y, groups, keys


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def top1_accuracy(keys: pd.DataFrame, y, p) -> float:
    """Per-point Top-1: for each GPS point pick the highest-scoring candidate.

    Only points with at least one true candidate are counted (points whose true arc
    is not among the candidates are a candidate-generation miss, Member 2's metric).
    """
    d = keys.reset_index(drop=True).rename(columns={"p": "pt"})
    d["y"] = np.asarray(y)
    d["score"] = np.asarray(p)
    cols = ["g", "pt"]
    has_pos = d.groupby(cols)["y"].transform("max") == 1
    d = d[has_pos]
    if d.empty:
        return float("nan")
    top = d.sort_values("score", ascending=False, kind="stable").drop_duplicates(cols)
    return float(top["y"].mean())


def score_proba(y, proba, keys) -> dict:
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    return {
        "f1_macro": float(f1_score(y, (proba >= 0.5).astype(int), average="macro", zero_division=0)),
        "logloss": float(log_loss(y, proba, labels=[0, 1])),
        "top1": top1_accuracy(keys, y, proba),
    }


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------
# CV + test evaluation
# ----------------------------------------------------------------------------
def cross_validate_model(factory, seed, X, y, groups, keys, n_splits=5) -> dict:
    """GroupKFold by trip. Returns mean of each metric and mean per-fold train time."""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    rows = []
    for tr, va in gkf.split(X, y, groups):
        model = factory(seed)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_s = time.perf_counter() - t0
        m = score_proba(y[va], _proba(model, X[va]), keys.iloc[va])
        m["train_time_s"] = fit_s
        rows.append(m)
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    f1s = [r["f1_macro"] for r in rows]
    out["f1_macro_std"] = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0  # sample std over folds
    return out


def evaluate_model(name, factory, seed, X, y, groups, keys, Xt, yt, keys_t, n_splits=5) -> dict:
    cv = cross_validate_model(factory, seed, X, y, groups, keys, n_splits)
    model = factory(seed)
    t0 = time.perf_counter()
    model.fit(X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba_t = _proba(model, Xt)
    infer_s = time.perf_counter() - t0
    te = score_proba(yt, proba_t, keys_t)
    return {
        "model": name,
        "cv_f1_macro": cv["f1_macro"],
        "cv_f1_macro_std": cv["f1_macro_std"],
        "cv_logloss": cv["logloss"],
        "cv_top1": cv["top1"],
        "cv_train_time_s_per_fold": cv["train_time_s"],
        "test_f1_macro": te["f1_macro"],
        "test_logloss": te["logloss"],
        "test_top1": te["top1"],
        "final_train_time_s": train_s,
        "test_inference_time_s": infer_s,
        "test_inference_ms_per_row": 1000 * infer_s / max(len(Xt), 1),
        "n_train_rows": len(X),
        "n_test_rows": len(Xt),
        "n_features": X.shape[1],
        "seed": seed,
    }


SPEC_COLUMNS = ["model", "cv_macro_f1", "cv_macro_f1_std", "test_top1_accuracy", "test_macro_f1",
                "test_log_loss", "inference_time_s", "train_time_s"]


def to_spec_row(r: dict) -> dict:
    """Map a detailed result to the required model_comparison.csv columns.

    train_time_s = fit time on the full train set; inference_time_s = time to predict the
    whole student_test set; cv_macro_f1_std = sample std (ddof=1) of macro F1 across CV folds.
    """
    return {
        "model": r["model"],
        "cv_macro_f1": r["cv_f1_macro"],
        "cv_macro_f1_std": r["cv_f1_macro_std"],
        "test_top1_accuracy": r["test_top1"],
        "test_macro_f1": r["test_f1_macro"],
        "test_log_loss": r["test_logloss"],
        "inference_time_s": r["test_inference_time_s"],
        "train_time_s": r["final_train_time_s"],
    }


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--out", default="results/model_comparison.csv")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help=f"comma-separated from {list(MODEL_FACTORIES)}")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--append", action="store_true", help="append to existing --out (e.g. Member 4's models)")
    a = ap.parse_args()

    np.random.seed(a.seed)
    feats, X, y, groups, keys, Xt, yt, keys_t = prepare_data(a)

    results = []
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        if name not in MODEL_FACTORIES:
            raise SystemExit(f"unknown model {name}")
        try:
            MODEL_FACTORIES[name](a.seed)  # catches missing optional packages early
        except ImportError as e:
            print(f"!! SKIPPING {name}: {e}")
            continue
        print(f"== {name}")
        r = evaluate_model(name, MODEL_FACTORIES[name], a.seed, X, y, groups, keys, Xt, yt, keys_t, a.n_splits)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        results.append(r)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detail_out = out.with_name(out.stem + "_detailed.csv")  # extra columns (CV log loss, CV Top-1, ...)
    for path, df in [(out, pd.DataFrame([to_spec_row(r) for r in results], columns=SPEC_COLUMNS)),
                     (detail_out, pd.DataFrame(results))]:
        if a.append and path.exists():
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"saved {path}")


if __name__ == "__main__":
    main()