
from pathlib import Path
import numpy as np
import pandas as pd

from pathsnap.candidates import (
    read_trip,
    project_coordinates
)
from pathsnap.hmm import viterbi_decode

trip_id = "00000001"
trip_dir = Path("train") / trip_id

track, nodes, arcs = read_trip(trip_dir)
_, node_xy, _ = project_coordinates(track, nodes)

candidates = pd.read_csv(
    f"results/features/train/{trip_id}_features.csv"
)

# Temporary distance-based scores for a smoke test.
# Replace these with trained-model probabilities later.
candidates["candidate_probability"] = np.exp(
    -candidates["distance_m"] / 20.0
)

# Independent closest-road baseline
baseline = (
    candidates.sort_values("distance_m")
    .drop_duplicates("observation_id")
    [["observation_id", "arc_id"]]
    .rename(columns={"arc_id": "baseline_arc_id"})
)

decoded = viterbi_decode(
    candidates,
    arcs,
    node_xy,
    probability_col="candidate_probability"
)

result = baseline.merge(
    decoded,
    on="observation_id",
    how="inner",
    validate="one_to_one"
)

result = result.sort_values(
    "observation_id"
).reset_index(drop=True)

print("GPS observations:", len(track))
print("Decoded observations:", len(result))
print(
    "Observations changed by decoder:",
    int((
        result["baseline_arc_id"]
        != result["predicted_arc_id"]
    ).sum())
)

Path("results").mkdir(exist_ok=True)

result.to_csv(
    "results/hmm_smoke_test.csv",
    index=False
)

print("Saved: results/hmm_smoke_test.csv")
print(result.head(10))
