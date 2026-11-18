"""
Optuna (TPE) hyperparameter tuning for LightGBM.

- 30+ trials, TPE sampler with a fixed seed
- Objective: mean macro F1 over GroupKFold(trip_id) on the TRAIN set only
- student_test is used only at the end, to compare tuned vs untuned
- Uses the SAME data loading and metric code as harness.py, so the
  untuned row matches results/model_comparison.csv

Saves:
    results/optuna_trials.csv
    results/optuna_history.png
    results/optuna_best_params.json
    results/tuned_vs_untuned.csv
    models/final_model.pkl
    models/final_model_features.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (  # noqa: E402
    DEFAULT_FEATURES,
    SEED,
    SPEC_COLUMNS,
    _lgbm,
    _proba,
    load_table,
    score_proba,
    split_xy,
)


def prepare(a, features):
    tr = load_table(a.train)
    te = load_table(a.test)
    if a.usable_col and a.usable_col in tr.columns:
        tr = tr[tr[a.usable_col] == 1].copy()
        te = te[te[a.usable_col] == 1].copy()
    X, y, groups, keys = split_xy(tr, features, a.label_col, a.group_col, a.point_col)
    Xt, yt, _, keys_t = split_xy(te, features, a.label_col, a.group_col, a.point_col)
    return X, y, groups, keys, Xt, yt, keys_t


def cv_macro_f1(factory, X, y, groups, keys, n_splits):
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    f1s = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model = factory()
        model.fit(X[tr_idx], y[tr_idx])
        m = score_proba(y[va_idx], _proba(model, X[va_idx]), keys.iloc[va_idx])
        f1s.append(m["f1_macro"])
    std = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
    return float(np.mean(f1s)), std


def make_lgbm(params, seed):
    return LGBMClassifier(
        **params, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1
    )


def suggest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def plot_history(study, path):
    done = [t for t in study.trials if t.value is not None]
    nums = [t.number for t in done]
    vals = [t.value for t in done]
    best = np.maximum.accumulate(vals)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(nums, vals, s=22, alpha=0.7, label="Trial CV macro F1")
    plt.plot(nums, best, color="tab:red", lw=2, label="Best so far")
    plt.xlabel("Trial")
    plt.ylabel("CV macro F1 (GroupKFold by trip_id, train only)")
    plt.title("Optuna TPE optimisation history (LightGBM)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="trip_id")
    ap.add_argument("--point-col", default="observation_id")
    ap.add_argument("--usable-col", default="label_usable")
    ap.add_argument("--feature-cols", default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    a = ap.parse_args()

    features = [f for f in a.feature_cols.split(",") if f]
    res_dir, mod_dir = Path(a.results_dir), Path(a.models_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed)

    X, y, groups, keys, Xt, yt, keys_t = prepare(a, features)
    print(f"Train rows: {len(y)}, test rows: {len(yt)}, features: {features}")

    # 1. Optuna TPE: uses TRAIN data only (GroupKFold by trip_id)
    def objective(trial):
        params = suggest(trial)
        mean_f1, std_f1 = cv_macro_f1(
            lambda: make_lgbm(params, a.seed), X, y, groups, keys, a.n_splits
        )
        trial.set_user_attr("cv_macro_f1_std", std_f1)
        return mean_f1

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=a.seed),
        study_name="lightgbm_tpe",
    )

    def log_cb(study, trial):
        print(
            f"Trial {trial.number + 1}/{a.n_trials}: "
            f"F1={trial.value:.4f}  best={study.best_value:.4f}"
        )

    study.optimize(objective, n_trials=a.n_trials, callbacks=[log_cb])

    study.trials_dataframe().to_csv(res_dir / "optuna_trials.csv", index=False)
    plot_history(study, res_dir / "optuna_history.png")
    best_params = study.best_params
    (res_dir / "optuna_best_params.json").write_text(
        json.dumps({"best_cv_macro_f1": study.best_value, "params": best_params}, indent=2)
    )
    print("Best CV macro F1:", round(study.best_value, 4))
    print("Best params:", best_params)

    # 2. Tuned vs untuned: same metrics as model_comparison.csv
    variants = {
        "lightgbm_untuned": lambda: _lgbm(a.seed),
        "lightgbm_tuned": lambda: make_lgbm(best_params, a.seed),
    }
    rows, fitted = [], {}
    for name, factory in variants.items():
        cv_mean, cv_std = cv_macro_f1(factory, X, y, groups, keys, a.n_splits)
        model = factory()
        t0 = time.perf_counter()
        model.fit(X, y)
        train_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = _proba(model, Xt)
        infer = time.perf_counter() - t0
        m = score_proba(yt, proba, keys_t)
        rows.append({
            "model": name,
            "cv_macro_f1": cv_mean,
            "cv_macro_f1_std": cv_std,
            "test_top1_accuracy": m["top1"],
            "test_macro_f1": m["f1_macro"],
            "test_log_loss": m["logloss"],
            "inference_time_s": infer,
            "train_time_s": train_time,
        })
        fitted[name] = model

    comp = pd.DataFrame(rows, columns=SPEC_COLUMNS)
    comp.to_csv(res_dir / "tuned_vs_untuned.csv", index=False)
    print(comp.to_string(index=False))

    # 3. Export tuned model + the feature list it expects
    final = fitted["lightgbm_tuned"]
    joblib.dump(final, mod_dir / "final_model.pkl")
    (mod_dir / "final_model_features.json").write_text(json.dumps({
        "feature_names": features,
        "classes": [int(c) for c in final.classes_],
        "positive_class": 1,
        "preprocessing": None,
        "params": best_params,
        "seed": a.seed,
        "note": "Pass columns in exactly this order. "
                "Use predict_proba(X)[:, 1] as the candidate score.",
    }, indent=2))
    print("Saved models/final_model.pkl and models/final_model_features.json")


if __name__ == "__main__":
    main()