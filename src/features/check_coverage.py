
from pathlib import Path
import pandas as pd

trip_id = "00000001"

candidates = pd.read_csv(
    f"results/features/{trip_id}_candidates.csv"
)

route = pd.read_csv(
    Path("train") / trip_id / f"{trip_id}.route",
    sep=r"\s+",
    header=None,
    names=["arc_id"]
)

reference_arcs = set(route["arc_id"].astype(int))

candidates["route_member"] = (
    candidates["arc_id"].isin(reference_arcs)
)

coverage = candidates.groupby("observation_id").agg(
    candidate_count=("arc_id", "size"),
    reference_arc_present=("route_member", "any"),
    nearest_distance_m=("distance_m", "min")
)

print("GPS observations:", len(coverage))
print(
    "Observations containing a reference-route arc:",
    int(coverage["reference_arc_present"].sum())
)
print(
    "Reference-route membership coverage:",
    round(coverage["reference_arc_present"].mean() * 100, 2),
    "%"
)
print(
    "Observations without reference-route candidates:",
    int((~coverage["reference_arc_present"]).sum())
)

print("\nNearest candidate distance:")
print(coverage["nearest_distance_m"].describe())
