
from pathlib import Path
from pathsnap.candidates import build_candidates

trip = Path("train/00000001")

df = build_candidates(
    trip,
    radius=50,
    max_candidates=12
)

Path("results/features").mkdir(
    parents=True, exist_ok=True
)

df.to_csv(
    "results/features/00000001_shared.csv",
    index=False
)

print(df.head())
print("\nMissing values:")
print(df.isna().sum())
print("\nCandidate counts:")
print(df.groupby("observation_id").size().describe())
print("\nFallback candidate rows:",
      df["outside_radius"].sum())
