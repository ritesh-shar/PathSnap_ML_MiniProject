
from pathlib import Path
from time import perf_counter
import pandas as pd

from pathsnap.candidates import read_trip, project_coordinates
from pathsnap.road_graph import (
    build_road_graph,
    network_transition_distance
)

trip_dir = Path("train/00000001")
track, nodes, arcs = read_trip(trip_dir)
_, node_xy, _ = project_coordinates(track, nodes)

graph = build_road_graph(arcs, node_xy)

candidates = pd.read_csv(
    "results/features/train/00000001_features.csv"
)

groups = [
    group for _, group in candidates.groupby(
        "observation_id", sort=True
    )
]

pairs = []

for previous, current in zip(groups[:-1], groups[1:]):
    for previous_arc in previous["arc_id"].head(3):
        for current_arc in current["arc_id"].head(3):
            pairs.append((int(previous_arc), int(current_arc)))
            if len(pairs) >= 100:
                break
        if len(pairs) >= 100:
            break
    if len(pairs) >= 100:
        break

start = perf_counter()

distances = [
    network_transition_distance(
        graph, a, b, arcs, node_xy
    )
    for a, b in pairs
]

elapsed = perf_counter() - start

print("Transition pairs tested:", len(pairs))
print("Total time (seconds):", round(elapsed, 4))
print("Average time per pair (ms):",
      round(elapsed / len(pairs) * 1000, 4))
print("Unreachable pairs:",
      sum(d == float("inf") for d in distances))
