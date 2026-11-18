
from pathlib import Path
import pandas as pd

records = []

for folder in sorted(Path("train").iterdir()):
    if not folder.is_dir():
        continue

    trip = folder.name

    arcs = pd.read_csv(
        folder / f"{trip}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    route = pd.read_csv(
        folder / f"{trip}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    route_ids = route["arc_id"].to_numpy(dtype=int)

    if len(route_ids) < 2:
        continue

    previous = arcs.iloc[route_ids[:-1]]
    following = arcs.iloc[route_ids[1:]]

    connected = (
        previous["to_node"].to_numpy()
        == following["from_node"].to_numpy()
    )

    records.append({
        "trip_id": trip,
        "route_transitions": len(connected),
        "directly_connected": int(connected.sum()),
        "direct_connection_breaks": int((~connected).sum()),
        "direct_connection_rate": float(connected.mean())
    })

result = pd.DataFrame(records)

Path("results").mkdir(exist_ok=True)

result.to_csv(
    "results/route_connectivity.csv",
    index=False
)

print("Trips analysed:", len(result))
print(
    "Total direct-connection breaks:",
    result["direct_connection_breaks"].sum()
)
print(
    "Overall direct connection rate:",
    result["directly_connected"].sum()
    / result["route_transitions"].sum()
)
print("\nTrips with most breaks:")
print(
    result.sort_values(
        "direct_connection_breaks",
        ascending=False
    ).head(10).to_string(index=False)
)
