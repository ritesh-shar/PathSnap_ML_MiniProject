
from pathlib import Path
import pandas as pd

for partition in ["train", "student_test"]:
    folder = Path("results/features") / partition

    # Excludes *_candidates.csv automatically
    files = sorted(folder.glob("*_features.csv"))

    if not files:
        raise FileNotFoundError(f"No feature files in {folder}")

    frames = [pd.read_csv(file) for file in files]
    combined = pd.concat(frames, ignore_index=True)

    output = Path("results") / f"{partition}_features.csv"
    combined.to_csv(output, index=False)

    print(f"\n{partition.upper()}")
    print(f"Trips: {len(files)}")
    print(f"Rows: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")
    print(f"Saved: {output}")
