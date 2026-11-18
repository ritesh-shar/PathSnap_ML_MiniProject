
"""Member 3 (after 11:30): Optuna TPE tuning, tuned-vs-untuned comparison, model export.
 
Usage:
    python training/tune.py --train features_train.csv --test features_test.csv
 
Picks the best model in results/model_comparison.csv by cv_f1_macro (or pass --model).
Tuning uses GroupKFold by trip_id on the TRAIN file only; student_test is only used for
the final tuned-vs-untuned comparison. Needs: pip install optuna lightgbm
Outputs:
    results/optuna_trials.csv, results/optuna_history.png,
    results/tuned_vs_untuned.csv, models/final_model.pkl, models/feature_list.json
"""
from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
 
 
def build_model(name: str, params: dict, seed: int):
    """Fresh estimator for `name` with tuned `params`."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
 
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
 
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, subsample_freq=1, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed, **params))
    raise SystemExit(f"tuning not defined for model '{name}' (use lightgbm, random_forest or logreg)")
 
 
def suggest_params(trial, name: str) -> dict:
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    raise SystemExit(f"tuning not defined for model '{name}'")
 
 
METRIC_KEY = {"f1_macro": "f1_macro", "top1": "top1", "logloss": "logloss"}
 
 
def make_objective(name, seed, X, y, groups, keys, n_splits, metric):
    def objective(trial):
        params = suggest_params(trial, name)
        factory = lambda s: build_model(name, params, s)  # noqa: E731
        cv = H.cross_validate_model(factory, seed, X, y, groups, keys, n_splits)
        return cv[METRIC_KEY[metric]]
 
    return objective
 
 
def plot_history(trials: pd.DataFrame, metric: str, direction: str, path: Path):
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    v = trials["value"].to_numpy()
    best = np.maximum.accumulate(v) if direction == "maximize" else np.minimum.accumulate(v)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(trials["number"], v, s=18, alpha=0.6, label="trial")
    ax.plot(trials["number"], best, color="#d62728", lw=2, label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"CV {metric}")
    ax.set_title("Optuna TPE optimisation history")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def main():
    ap = argparse.ArgumentParser()
    H.add_data_args(ap)
    ap.add_argument("--comparison", default="results/model_comparison.csv")
    ap.add_argument("--model", default=None, help="model to tune; default = best cv_f1_macro in comparison csv")
    ap.add_argument("--metric", default="f1_macro", choices=list(METRIC_KEY))
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=H.SEED)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    a = ap.parse_args()
 
    import optuna  # lazy so the helpers above import without optuna
 
    name = a.model
    if name is None:
        comp = pd.read_csv(a.comparison)
        comp = comp[comp["model"].isin(["lightgbm", "random_forest", "logreg"])]
        name = comp.sort_values("cv_macro_f1", ascending=False).iloc[0]["model"]
    print(f"tuning: {name}")
 
    feats, X, y, groups, keys, Xt, yt, keys_t = H.prepare_data(a)
 
    direction = "minimize" if a.metric == "logloss" else "maximize"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=a.seed))
    study.optimize(make_objective(name, a.seed, X, y, groups, keys, a.n_splits, a.metric),
                   n_trials=a.n_trials, show_progress_bar=False)
    print("best value:", study.best_value, "params:", study.best_params)
 
    rdir, mdir = Path(a.results_dir), Path(a.models_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe()
    trials.to_csv(rdir / "optuna_trials.csv", index=False)
    plot_history(trials, a.metric, direction, rdir / "optuna_history.png")
 
    # untuned vs tuned, same metrics, same seed
    untuned = H.evaluate_model(f"{name} (untuned)", H.MODEL_FACTORIES[name], a.seed, X, y, groups, keys,
                               Xt, yt, keys_t, a.n_splits)
    best = study.best_params
    tuned = H.evaluate_model(f"{name} (tuned)", lambda s: build_model(name, best, s), a.seed, X, y, groups,
                             keys, Xt, yt, keys_t, a.n_splits)
    pd.DataFrame([untuned, tuned]).to_csv(rdir / "tuned_vs_untuned.csv", index=False)
 
    # export: tuned model fitted on train only, plus the feature list it expects
    final = build_model(name, best, a.seed)
    final.fit(X, y)
    joblib.dump({"model": final, "features": feats, "model_name": name, "params": best,
                 "seed": a.seed, "label_col": a.label_col}, mdir / "final_model.pkl")
    (mdir / "feature_list.json").write_text(json.dumps(feats, indent=2))
    print(f"saved {rdir/'optuna_trials.csv'}, optuna_history.png, tuned_vs_untuned.csv, "
          f"{mdir/'final_model.pkl'}")
 
 
if __name__ == "__main__":
    main()
 