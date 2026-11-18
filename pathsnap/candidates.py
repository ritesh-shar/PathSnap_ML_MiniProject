
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


FEATURE_COLUMNS = [
    "distance_m",
    "segment_length_m",
    "heading_difference_rad",
    "gps_speed_mps",
    "time_gap_s",
]


def read_trip(trip_dir):
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    def read(ext, names):
        return pd.read_csv(
            trip_dir / f"{trip_id}.{ext}",
            sep=r"\s+",
            header=None,
            names=names
        )

    track = read(
        "track", ["longitude", "latitude", "timestamp"]
    )
    nodes = read(
        "nodes", ["longitude", "latitude"]
    )
    arcs = read(
        "arcs", ["from_node", "to_node"]
    )

    return track, nodes, arcs


def project_coordinates(track, nodes):
    lon = float(nodes["longitude"].median())
    lat = float(nodes["latitude"].median())

    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    epsg = (32600 if lat >= 0 else 32700) + zone

    crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )

    def convert(df):
        x, y = transformer.transform(
            df["longitude"].to_numpy(),
            df["latitude"].to_numpy()
        )
        return np.column_stack([x, y])

    return convert(track), convert(nodes), epsg


def build_candidates(trip_dir, radius=50, max_candidates=12):
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track, nodes, arcs = read_trip(trip_dir)
    gps_xy, node_xy, epsg = project_coordinates(track, nodes)

    starts = arcs["from_node"].to_numpy(dtype=int)
    ends = arcs["to_node"].to_numpy(dtype=int)

    if not (
        np.all((starts >= 0) & (starts < len(nodes))) and
        np.all((ends >= 0) & (ends < len(nodes)))
    ):
        raise ValueError(f"Invalid node IDs in trip {trip_id}")

    segments = [
        LineString([node_xy[u], node_xy[v]])
        for u, v in zip(starts, ends)
    ]

    tree = STRtree(segments)

    times = track["timestamp"].to_numpy(dtype=float)
    movement = np.diff(gps_xy, axis=0)
    dt = np.diff(times)

    rows = []

    for i, xy in enumerate(gps_xy):
        point = Point(float(xy[0]), float(xy[1]))

        nearby = tree.query(point.buffer(radius))
        candidates = []

        if i > 0 and dt[i - 1] > 0:
            speed = float(
                np.linalg.norm(movement[i - 1]) / dt[i - 1]
            )
            heading = float(
                np.arctan2(
                    movement[i - 1, 1],
                    movement[i - 1, 0]
                )
            )
            time_gap = float(dt[i - 1])
        else:
            speed = 0.0
            heading = None
            time_gap = 0.0

        for arc_id in nearby:
            arc_id = int(arc_id)
            segment = segments[arc_id]
            distance = float(point.distance(segment))

            if distance > radius:
                continue

            u, v = starts[arc_id], ends[arc_id]
            vec = node_xy[v] - node_xy[u]

            road_heading = float(
                np.arctan2(vec[1], vec[0])
            )

            if heading is None or speed < 0.5:
                difference = 0.0
            else:
                difference = float(
                    np.abs(np.arctan2(
                        np.sin(heading - road_heading),
                        np.cos(heading - road_heading)
                    ))
                )

            candidates.append({
                "trip_id": trip_id,
                "observation_id": i,
                "arc_id": arc_id,
                "distance_m": distance,
                "segment_length_m": float(segment.length),
                "heading_difference_rad": difference,
                "gps_speed_mps": speed,
                "time_gap_s": time_gap,
                "timestamp": float(times[i]),
                "projection_epsg": epsg
            })

        candidates.sort(
            key=lambda row: (row["distance_m"], row["arc_id"])
        )

        # Retain nearest segment if no arc is within radius.
        # Mark it so failures are not hidden.
        if not candidates:
            nearest_id = int(tree.nearest(point))
            segment = segments[nearest_id]
            u, v = starts[nearest_id], ends[nearest_id]
            vec = node_xy[v] - node_xy[u]
            road_heading = float(np.arctan2(vec[1], vec[0]))
            if heading is None or speed < 0.5:
                difference = 0.0
            else:
                difference = float(abs(np.arctan2(
                    np.sin(heading-road_heading),
                    np.cos(heading-road_heading)
                )))
            candidates = [{
                "trip_id": trip_id,
                "observation_id": i,
                "arc_id": nearest_id,
                "distance_m": float(point.distance(segment)),
                "segment_length_m": float(segment.length),
                "heading_difference_rad": difference,
                "gps_speed_mps": speed,
                "time_gap_s": time_gap,
                "timestamp": float(times[i]),
                "projection_epsg": epsg
            }]

        for row in candidates[:max_candidates]:
            row["outside_radius"] = int(
                row["distance_m"] > radius
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    print(
        f"{trip_id}: {len(track)} GPS points, "
        f"{len(result)} candidates, EPSG:{epsg}"
    )
    return result
