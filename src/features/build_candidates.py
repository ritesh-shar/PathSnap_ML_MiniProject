
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TRIP_ID = "00000001"
BASE = Path("train") / TRIP_ID
OUT = Path("results/features")
OUT.mkdir(parents=True, exist_ok=True)

def load_file(extension, columns):
    return pd.read_csv(
        BASE / f"{TRIP_ID}.{extension}",
        sep=r"\s+",
        header=None,
        names=columns
    )

track = load_file(
    "track", ["longitude", "latitude", "timestamp"]
)
nodes = load_file(
    "nodes", ["longitude", "latitude"]
)
arcs = load_file(
    "arcs", ["from_node", "to_node"]
)
route = load_file("route", ["arc_id"])

# Validate node references
valid = (
    arcs["from_node"].between(0, len(nodes)-1) &
    arcs["to_node"].between(0, len(nodes)-1)
)
assert valid.all(), "Invalid road-network node IDs"

# Approximate local coordinates in metres.
# Suitable for local-scale exploration, not global routes.
earth_radius = 6371000.0
lat0 = np.deg2rad(
    pd.concat([track["latitude"], nodes["latitude"]]).mean()
)
lon0 = np.deg2rad(nodes["longitude"].mean())

def to_xy(df):
    lon = np.deg2rad(df["longitude"].to_numpy())
    lat = np.deg2rad(df["latitude"].to_numpy())

    x = earth_radius * (lon - lon0) * np.cos(lat0)
    y = earth_radius * (lat - lat0)
    return np.column_stack([x, y])

node_xy = to_xy(nodes)
gps_xy = to_xy(track)

starts = arcs["from_node"].to_numpy(dtype=int)
ends = arcs["to_node"].to_numpy(dtype=int)

# Spatial index of road-network nodes
tree = cKDTree(node_xy)

# Build node-to-incident-arc lookup
incident = [[] for _ in range(len(nodes))]
for arc_id, (u, v) in enumerate(zip(starts, ends)):
    incident[u].append(arc_id)
    incident[v].append(arc_id)

# Candidate search configuration
K_NEAREST_NODES = 12
MAX_CANDIDATES = 8

# Nearest nodes are used to propose nearby arcs.
_, nearest_nodes = tree.query(
    gps_xy, k=min(K_NEAREST_NODES, len(nodes))
)
nearest_nodes = np.atleast_2d(nearest_nodes)

if nearest_nodes.shape[0] != len(gps_xy):
    nearest_nodes = nearest_nodes.T

reference_arcs = set(route["arc_id"].astype(int))
rows = []

for obs_id, point in enumerate(gps_xy):
    candidate_ids = set()

    for node_id in np.ravel(nearest_nodes[obs_id]):
        candidate_ids.update(incident[int(node_id)])

    scored = []

    for arc_id in candidate_ids:
        a = node_xy[starts[arc_id]]
        b = node_xy[ends[arc_id]]

        vector = b - a
        length_sq = float(np.dot(vector, vector))

        if length_sq == 0:
            t = 0.0
        else:
            t = float(np.clip(
                np.dot(point-a, vector)/length_sq,
                0, 1
            ))

        projected = a + t*vector
        distance = float(np.linalg.norm(point-projected))
        segment_length = float(np.sqrt(length_sq))

        scored.append((
            distance, arc_id, segment_length,
            float(np.arctan2(vector[1], vector[0]))
        ))

    scored.sort()

    # Keep the closest candidate roads
    for distance, arc_id, length, bearing in scored[:MAX_CANDIDATES]:
        rows.append({
            "trip_id": TRIP_ID,
            "observation_id": obs_id,
            "arc_id": arc_id,
            "distance_m": distance,
            "segment_length_m": length,
            "road_bearing_rad": bearing,
            "timestamp": track.iloc[obs_id]["timestamp"],
            # Route membership is NOT a GPS-level ground-truth label.
            "in_reference_route": int(arc_id in reference_arcs)
        })

features = pd.DataFrame(rows)

output_path = OUT / f"{TRIP_ID}_candidates.csv"
features.to_csv(output_path, index=False)

print("Trip:", TRIP_ID)
print("GPS observations:", len(track))
print("Generated candidate rows:", len(features))
print("Unique road arcs:", features["arc_id"].nunique())
print("Reference-route member candidates:",
      features["in_reference_route"].sum())
print("Saved:", output_path)
print(features.head(10).to_string(index=False))
