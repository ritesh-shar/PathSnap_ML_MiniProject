
from pathlib import Path
import networkx as nx

from pathsnap.candidates import (
    read_trip,
    project_coordinates
)
from pathsnap.road_graph import (
    build_road_graph,
    network_transition_distance
)

trip_dir = Path("train/00000001")

track, nodes, arcs = read_trip(trip_dir)
_, node_xy, _ = project_coordinates(track, nodes)

graph = build_road_graph(arcs, node_xy)

print("Road-network nodes:", graph.number_of_nodes())
print("Road-network edges:", graph.number_of_edges())

components = nx.number_weakly_connected_components(graph)
print("Weakly connected components:", components)

route_path = trip_dir / "00000001.route"

import pandas as pd

route = pd.read_csv(
    route_path,
    sep=r"\s+",
    header=None,
    names=["arc_id"]
)

route_ids = route["arc_id"].to_numpy(dtype=int)

for previous_arc, current_arc in zip(
    route_ids[:5], route_ids[1:6]
):
    distance = network_transition_distance(
        graph,
        previous_arc,
        current_arc,
        arcs,
        node_xy
    )
    print(
        f"Arc {previous_arc} -> {current_arc}: "
        f"{distance:.2f} metres"
    )
