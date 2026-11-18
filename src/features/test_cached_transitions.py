
from pathlib import Path
from time import perf_counter

from pathsnap.candidates import (
    read_trip,
    project_coordinates
)
from pathsnap.road_graph import (
    build_road_graph,
    make_cached_distance
)

trip = Path("train/00000001")

track, nodes, arcs = read_trip(trip)
_, node_xy, _ = project_coordinates(track, nodes)

graph = build_road_graph(arcs, node_xy)
distance = make_cached_distance(graph)

starts = arcs["from_node"].to_numpy(dtype=int)
ends = arcs["to_node"].to_numpy(dtype=int)

pairs = [
    (2641, 64973),
    (64973, 133948),
    (133948, 133949),
    (133949, 139208),
    (139208, 133950)
]

for a, b in pairs:
    source = ends[a]
    target = starts[b]

    start = perf_counter()
    result = distance(source, target)
    elapsed = perf_counter() - start

    print(
        f"{a} -> {b}: {result:.2f} m, "
        f"{elapsed * 1000:.3f} ms"
    )

print("Cache:", distance.cache_info())
