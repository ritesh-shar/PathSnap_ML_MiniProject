"""PathSnap exploratory data analysis (train and student_test only).

Usage:
    python run_eda.py "<path to Student folder>" results/eda

Outputs (in the output folder):
    trip_inventory.csv     one row per trip with all the statistics below
    missing_values.csv     missing or invalid values per file type and split
    split_summary.csv      the inventory summarised per split
    *.png                  plots for the report
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pathsnap_io import list_trips, load_trip, to_xy, point_to_segments_dist

SPLITS = ["train", "student_test"]   # evaluator-testing data is never touched here
GAP_S = 5                            # a time step above this counts as a GPS gap


def trip_stats(trip_dir, split):
    d = load_trip(trip_dir)
    tid = d["trip_id"]
    tr, nd, ar, rt = d["track"], d["nodes"], d["arcs"], d["route"]
    row = {"split": split, "trip_id": tid}

    # file availability and size
    for ext in ["track", "nodes", "arcs", "route"]:
        f = Path(trip_dir) / f"{tid}.{ext}"
        row[f"has_{ext}"] = f.exists()
        row[f"kb_{ext}"] = round(f.stat().st_size / 1024, 1) if f.exists() else np.nan

    # GPS track
    row["n_points"] = len(tr)
    row["track_nan"] = int(tr.isna().sum().sum())
    row["track_bad_coord"] = int(((tr.lon.abs() > 180) | (tr.lat.abs() > 90)).sum())
    dt = np.diff(tr.t.values)
    row["duration_s"] = tr.t.iloc[-1] - tr.t.iloc[0]
    row["dt_nonpositive"] = int((dt <= 0).sum())        # out-of-order or duplicate timestamps
    row["pct_dt_1s"] = round(100 * (dt == 1).mean(), 2)
    row["dt_median"] = float(np.median(dt))
    row["dt_max"] = float(dt.max())
    row["n_gaps"] = int((dt > GAP_S).sum())
    x, y = to_xy(tr.lon, tr.lat)
    step = np.hypot(np.diff(x), np.diff(y))
    speed = step / np.where(dt > 0, dt, np.nan)
    row["track_km"] = round(step.sum() / 1000, 2)
    row["speed_median_ms"] = round(float(np.nanmedian(speed)), 2)
    row["speed_max_ms"] = round(float(np.nanmax(speed)), 2)
    row["n_speed_gt_50ms"] = int((speed > 50).sum())     # > 180 km/h suggests a GPS jump
    row["pct_stationary"] = round(100 * (step < 0.5).mean(), 2)
    row["lon_min"], row["lon_max"] = tr.lon.min(), tr.lon.max()
    row["lat_min"], row["lat_max"] = tr.lat.min(), tr.lat.max()
    row["bbox_km"] = round(max(x.max() - x.min(), y.max() - y.min()) / 1000, 2)

    # road network
    row["n_nodes"], row["n_arcs"] = len(nd), len(ar)
    row["arcs_bad_node_id"] = int(((ar.u < 0) | (ar.v < 0) | (ar.u >= len(nd)) | (ar.v >= len(nd))).sum())
    row["self_loops"] = int((ar.u == ar.v).sum())
    pairs = set(zip(ar.u, ar.v))
    row["pct_two_way"] = round(100 * np.mean([(v, u) in pairs for u, v in pairs]), 1)
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nd)))
    G.add_edges_from(zip(ar.u, ar.v))
    comps = list(nx.weakly_connected_components(G))
    row["n_components"] = len(comps)
    row["largest_comp_pct"] = round(100 * max(len(c) for c in comps) / len(nd), 2)
    row["isolated_nodes"] = sum(1 for n in G if G.degree(n) == 0)
    row["n_strong_components"] = nx.number_strongly_connected_components(G)

    # reference route
    if rt is not None:
        r = rt.arc_id.values
        row["route_len"] = len(r)
        row["route_bad_arc_id"] = int(((r < 0) | (r >= len(ar))).sum())
        row["route_repeats"] = len(r) - len(set(r))
        u, v = ar.u.values, ar.v.values
        row["route_breaks"] = int(sum(v[r[k]] != u[r[k + 1]] for k in range(len(r) - 1)))
        # GPS noise: distance from each GPS point to the nearest arc on the reference route
        nx_, ny_ = to_xy(nd.lon, nd.lat, tr.lon, tr.lat)
        dist = point_to_segments_dist(x, y, nx_[u[r]], ny_[u[r]], nx_[v[r]], ny_[v[r]])
        row["gps_to_route_median_m"] = round(float(np.median(dist)), 2)
        row["gps_to_route_p95_m"] = round(float(np.percentile(dist, 95)), 2)
        row["gps_to_route_max_m"] = round(float(dist.max()), 2)
    return row, dt, speed


def main(data_root, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, all_dt, all_speed = [], {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    for split in SPLITS:
        for td in list_trips(Path(data_root) / split):
            row, dt, sp = trip_stats(td, split)
            rows.append(row)
            all_dt[split].append(dt)
            all_speed[split].append(sp)
            print(f"{split:12s} {row['trip_id']}  points={row['n_points']:6d}  nodes={row['n_nodes']:6d}")
    inv = pd.DataFrame(rows)
    inv.to_csv(out / "trip_inventory.csv", index=False)

    # missing / invalid value table
    miss = inv.groupby("split")[["track_nan", "track_bad_coord", "dt_nonpositive", "arcs_bad_node_id",
                                 "self_loops", "route_bad_arc_id", "route_repeats", "route_breaks",
                                 "isolated_nodes"]].sum()
    miss["trips_missing_a_file"] = inv.assign(m=~inv[[c for c in inv if c.startswith("has_")]].all(axis=1)) \
        .groupby("split")["m"].sum()
    miss.to_csv(out / "missing_values.csv")

    summ = inv.groupby("split")[["n_points", "duration_s", "dt_max", "n_gaps", "track_km", "n_nodes",
                                 "n_arcs", "n_components", "route_len", "route_breaks",
                                 "gps_to_route_median_m", "gps_to_route_p95_m"]].describe().T
    summ.to_csv(out / "split_summary.csv")

    colors = {"train": "#1f77b4", "student_test": "#ff7f0e"}

    # 1. GPS points per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        ax.hist(inv[inv.split == s].n_points, bins=30, alpha=0.6, label=s, color=colors[s])
    ax.set(xlabel="GPS observations per trip", ylabel="Number of trips", title="Trip length (GPS points)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "01_points_per_trip.png", dpi=150); plt.close(fig)

    # 2. sampling interval distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        d = np.concatenate(all_dt[s])
        ax.hist(np.clip(d, 0, 30), bins=30, alpha=0.6, label=s, color=colors[s], log=True)
    ax.set(xlabel="Time between consecutive GPS points (s, values above 30 shown at 30)",
           ylabel="Count (log scale)", title="Sampling interval")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "02_sampling_interval.png", dpi=150); plt.close(fig)

    # 3. largest GPS gap per trip
    fig, ax = plt.subplots(figsize=(9, 4))
    inv_sorted = inv.sort_values("dt_max", ascending=False)
    ax.bar(range(len(inv_sorted)), inv_sorted.dt_max, color=[colors[s] for s in inv_sorted.split])
    ax.axhline(GAP_S, color="red", ls="--", lw=1, label=f"gap threshold {GAP_S}s")
    ax.set(xlabel="Trips (sorted)", ylabel="Largest time gap (s)", title="Largest GPS gap per trip", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "03_max_gap_per_trip.png", dpi=150); plt.close(fig)

    # 4. road network size
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        t = inv[inv.split == s]
        ax.scatter(t.n_nodes, t.n_arcs, label=s, alpha=0.7, color=colors[s])
    ax.set(xscale="log", yscale="log", xlabel="Nodes", ylabel="Arcs (road segments)", title="Road network size per trip")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "04_network_size.png", dpi=150); plt.close(fig)

    # 5. GPS noise
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        ax.hist(inv[inv.split == s].gps_to_route_p95_m, bins=25, alpha=0.6, label=s, color=colors[s])
    ax.set(xlabel="95th percentile distance from GPS point to reference route (m)", ylabel="Trips",
           title="GPS noise relative to the reference route")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "05_gps_noise.png", dpi=150); plt.close(fig)

    # 6. speed distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        sp = np.concatenate(all_speed[s])
        sp = sp[np.isfinite(sp)]
        ax.hist(np.clip(sp * 3.6, 0, 200), bins=50, alpha=0.6, label=s, color=colors[s], log=True)
    ax.set(xlabel="Speed between consecutive points (km/h, clipped at 200)", ylabel="Count (log)",
           title="Implied vehicle speed")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "06_speed.png", dpi=150); plt.close(fig)

    # 7. where the trips are (trips are spread worldwide, so plot one marker per trip)
    inv["lon_c"] = (inv.lon_min + inv.lon_max) / 2
    inv["lat_c"] = (inv.lat_min + inv.lat_max) / 2
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in SPLITS:
        t = inv[inv.split == s]
        ax.scatter(t.lon_c, t.lat_c, s=25, alpha=0.8, label=f"{s} ({len(t)} trips)", color=colors[s])
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xlabel="Longitude", ylabel="Latitude",
           title="Location of each trip (trip centre)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "07_trip_locations.png", dpi=150); plt.close(fig)

    # 8. the same as an interactive Folium world map
    import folium
    wm = folium.Map(location=[30, 0], zoom_start=2, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri streets").add_to(wm)
    for s in SPLITS:
        fg = folium.FeatureGroup(name=s)
        for r in inv[inv.split == s].itertuples():
            folium.CircleMarker([r.lat_c, r.lon_c], radius=5, color=colors[s], fill=True, fill_opacity=0.9,
                                tooltip=f"{s} trip {r.trip_id}: {r.n_points} points, {r.track_km} km").add_to(fg)
        fg.add_to(wm)
    folium.LayerControl(collapsed=False).add_to(wm)
    wm.save(out / "08_trip_locations_map.html")
    inv.to_csv(out / "trip_inventory.csv", index=False)

    print("\nMissing / invalid values:\n", miss.to_string())
    print(f"\nSaved inventory, tables and plots to {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results/eda")
