"""Bar charts comparing the trained models, read from Member 3's results CSV.

Run from the repo root:
    python3 eda/plot_model_comparison.py results/model_comparison.csv results/plots

The CSV needs a model-name column plus any of the metric columns below. Column names are
matched loosely (case and underscores ignored), and metrics that are missing are skipped.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# (possible column names, chart title, y label, higher is better)
METRICS = [
    (["cv_macro_f1", "cv_f1", "cv_f1_macro", "cv_macro_f1_mean"], "Cross-validation Macro F1 (train, GroupKFold)", "Macro F1", True),
    (["test_top1_accuracy", "top1_accuracy", "student_test_top1", "test_top1", "top1"], "Student-test Top-1 accuracy", "Top-1 accuracy", True),
    (["test_macro_f1", "student_test_macro_f1", "test_f1", "macro_f1"], "Student-test Macro F1", "Macro F1", True),
    (["test_log_loss", "log_loss", "student_test_log_loss", "logloss"], "Student-test log loss", "Log loss", False),
    (["route_continuity", "continuity"], "Route continuity", "Route continuity", True),
    (["inference_time_s", "inference_time", "infer_time_s"], "Inference time", "Seconds", False),
    (["train_time_s", "training_time_s", "train_time", "fit_time_s"], "Training time", "Seconds", False),
]
NAME_COLS = ["model", "model_name", "name", "algorithm"]
STD_SUFFIXES = ["_std", "_sd"]


def norm(c):
    return c.lower().replace(" ", "_").replace("-", "_")


def find(df, options):
    cols = {norm(c): c for c in df.columns}
    for o in options:
        if o in cols:
            return cols[o]
    return None


def main(csv, out_dir):
    df = pd.read_csv(csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name_col = find(df, NAME_COLS) or df.columns[0]
    names = df[name_col].astype(str)

    found = []
    for options, title, ylabel, higher in METRICS:
        col = find(df, options)
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            continue
        std_col = next((c for c in df.columns for s in STD_SUFFIXES if norm(c) == norm(col) + s), None)
        err = pd.to_numeric(df[std_col], errors="coerce") if std_col else None

        best = vals.idxmax() if higher else vals.idxmin()
        colours = ["#d62728" if i == best else "#4c72b0" for i in range(len(df))]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, vals.fillna(0), yerr=err, capsize=4, color=colours)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "N/A" if pd.isna(v) else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set(title=f"{title} ({'higher' if higher else 'lower'} is better, best in red)", ylabel=ylabel)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        f = out / f"compare_{norm(col)}.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        found.append(col)
        print("Saved", f)

    # one combined chart of the main quality metrics
    quality = [find(df, m[0]) for m in METRICS[:3]]
    quality = [q for q in quality if q]
    if len(quality) >= 2:
        ax = df.set_index(name_col)[quality].plot.bar(figsize=(9, 4.5), rot=20)
        ax.set(title="Model comparison", ylabel="Score", ylim=(0, 1.05))
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out / "compare_all_quality.png", dpi=150)
        plt.close()
        print("Saved", out / "compare_all_quality.png")

    if not found:
        print("No metric columns recognised. Columns in the CSV are:", list(df.columns))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results/plots")
