"""Run harness.py unchanged, but with a smaller (faster) random forest.

Usage: same arguments as harness.py, e.g.
    python training/run_fast_rf.py --train "<train glob>" --test "<test glob>"

Random forest here: 60 trees, max depth 15, min 5 samples per leaf (state this in the report).
harness.py itself is not modified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


def fast_rf(seed):
    return RandomForestClassifier(n_estimators=60, max_depth=15, min_samples_leaf=5,
                                  n_jobs=-1, random_state=seed)


H.MODEL_FACTORIES["random_forest"] = fast_rf

if __name__ == "__main__":
    H.main()