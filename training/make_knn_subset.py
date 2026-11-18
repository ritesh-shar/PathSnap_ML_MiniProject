from pathlib import Path
import pandas as pd

df = pd.read_csv("results/train_features.csv")

trip_ids = sorted(df["trip_id"].unique())
selected = pd.Series(trip_ids).sample(n=10, random_state=42).tolist()

subset = df[df["trip_id"].isin(selected)].copy()
subset.to_csv("results/knn_train_subset.csv", index=False)

print("Selected trips:", selected)
print("Training rows:", len(subset))
print("Saved: results/knn_train_subset.csv")
