
import numpy as np
import networkx as nx


def build_road_graph(arcs, node_xy):
    """
    Build a directed road graph.

    Each road arc connects from_node to to_node.
    Edge weights are physical road-segment lengths.
    """
    graph = nx.DiGraph()

    starts = arcs["from_node"].to_numpy(dtype=int)
    ends = arcs["to_node"].to_numpy(dtype=int)

    for arc_id, (u, v) in enumerate(zip(starts, ends)):
        distance = float(
            np.linalg.norm(node_xy[v] - node_xy[u])
        )

        # DiGraph stores the shortest weight when
        # multiple arcs share the same endpoints.
        if graph.has_edge(int(u), int(v)):
            if distance < graph[u][v]["weight"]:
                graph[u][v]["weight"] = distance
        else:
            graph.add_edge(
                int(u), int(v), weight=distance
            )

    return graph


def network_transition_distance(
    graph, previous_arc, current_arc,
    arcs, node_xy
):
    """
    Estimate movement distance along the directed network
    between consecutive candidate road arcs.
    """
    previous_arc = int(previous_arc)
    current_arc = int(current_arc)

    if previous_arc == current_arc:
        return 0.0

    starts = arcs["from_node"].to_numpy(dtype=int)
    ends = arcs["to_node"].to_numpy(dtype=int)

    source = int(ends[previous_arc])
    target = int(starts[current_arc])

    if source == target:
        return 0.0

    try:
        return float(
            nx.shortest_path_length(
                graph,
                source=source,
                target=target,
                weight="weight"
            )
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")

from functools import lru_cache
import networkx as nx


def make_cached_distance(graph, cutoff_m=2000.0):
    """
    Return a cached directed-network distance function.

    Paths longer than cutoff_m are treated as unreachable
    within the configured local search distance.
    """

    @lru_cache(maxsize=20000)
    def distance(source, target):
        source = int(source)
        target = int(target)

        if source == target:
            return 0.0

        if source not in graph or target not in graph:
            return float("inf")

        lengths = nx.single_source_dijkstra_path_length(
            graph,
            source,
            cutoff=cutoff_m,
            weight="weight"
        )

        return float(lengths.get(target, float("inf")))

    return distance
