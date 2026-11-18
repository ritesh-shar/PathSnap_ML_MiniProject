
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from pathsnap.candidates import read_trip, project_coordinates


def align_reference_route(
    trip_dir,
    lookahead=80,
    max_label_distance_m=30
):
    """Create provisional GPS-to-route labels using forward alignment."""

    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track, nodes, arcs = read_trip(trip_dir)
    gps_xy, node_xy, _ = project_coordinates(track, nodes)

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    route_ids = route["arc_id"].to_numpy(dtype=int)
    starts = arcs["from_node"].to_numpy(dtype=int)
    ends = arcs["to_node"].to_numpy(dtype=int)

    if len(route_ids) == 0:
        raise ValueError(f"Empty reference route: {trip_id}")

    if not np.all(
        (route_ids >= 0) & (route_ids < len(arcs))
    ):
        raise ValueError(f"Invalid route arc IDs: {trip_id}")

    route_segments = [
        LineString([
            node_xy[starts[arc_id]],
            node_xy[ends[arc_id]]
        ])
        for arc_id in route_ids
    ]

    current_position = 0
    records = []

    for obs_id, coordinates in enumerate(gps_xy):
        point = Point(
            float(coordinates[0]),
            float(coordinates[1])
        )

        end = min(
            len(route_ids),
            current_position + lookahead
        )

        nearest_position = min(
            range(current_position, end),
            key=lambda position: point.distance(
                route_segments[position]
            )
        )

        alignment_distance = float(
            point.distance(route_segments[nearest_position])
        )

        current_position = nearest_position

        records.append({
            "trip_id": trip_id,
            "observation_id": obs_id,
            "reference_arc_id": int(
                route_ids[nearest_position]
            ),
            "reference_route_position": nearest_position,
            "alignment_distance_m": alignment_distance,
            "label_usable": int(
                alignment_distance <= max_label_distance_m
            )
        })

    return pd.DataFrame(records)
