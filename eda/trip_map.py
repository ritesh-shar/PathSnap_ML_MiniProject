"""Folium map of one trip: GPS points, nearby road network, reference route and (later) predicted route.

Usage:
    python trip_map.py "<path to Student folder>/train/00000048" results/maps

From other code (Member 4's UI, the report):
    from trip_map import make_trip_map
    m = make_trip_map(trip_dir, predicted_arcs=[571, 569, ...])
    m.save("trip.html")
"""
import sys
from pathlib import Path

import folium
import numpy as np
from branca.element import Element

from pathsnap_io import load_trip

BUFFER_DEG = 0.003   # about 300 m: only draw road segments near the trajectory


def _arc_lines(nodes, arcs, arc_ids):
    """Turn arc ids into [[lat, lon], [lat, lon]] line pairs for folium."""
    u, v = arcs.u.values[arc_ids], arcs.v.values[arc_ids]
    return [[[nodes.lat[a], nodes.lon[a]], [nodes.lat[b], nodes.lon[b]]] for a, b in zip(u, v)]


def make_trip_map(trip_dir, predicted_arcs=None, show_reference=True, max_points=1500):
    d = load_trip(trip_dir)
    tr, nd, ar, rt = d["track"], d["nodes"], d["arcs"], d["route"]

    # map centre and extent from the GPS track
    lat0, lat1 = tr.lat.min() - BUFFER_DEG, tr.lat.max() + BUFFER_DEG
    lon0, lon1 = tr.lon.min() - BUFFER_DEG, tr.lon.max() + BUFFER_DEG
    # OpenStreetMap blocks tiles when the HTML is opened as a local file, so Esri street tiles
    # are the default background and OpenStreetMap is kept as a second option.
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri streets").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
    m.fit_bounds([[lat0, lon0], [lat1, lon1]])

    # 1. road network, clipped to the trajectory area so big networks stay light
    inside = nd.lat.between(lat0, lat1) & nd.lon.between(lon0, lon1)
    keep = np.where(inside.values[ar.u.values] | inside.values[ar.v.values])[0]
    net = folium.FeatureGroup(name=f"Road network ({len(keep)} segments)")
    folium.PolyLine(_arc_lines(nd, ar, keep), color="#9e9e9e", weight=1.5, opacity=0.7).add_to(net)
    net.add_to(m)

    # 2. reference route
    if show_reference and rt is not None:
        ref = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(_arc_lines(nd, ar, rt.arc_id.values), color="#2e7d32", weight=6, opacity=0.8,
                        tooltip="Reference route").add_to(ref)
        ref.add_to(m)

    # 3. predicted route
    if predicted_arcs is not None and len(predicted_arcs):
        p = np.asarray(predicted_arcs)
        p = p[np.r_[True, p[1:] != p[:-1]]]          # drop consecutive repeats
        pred = folium.FeatureGroup(name="Predicted route")
        folium.PolyLine(_arc_lines(nd, ar, p), color="#d32f2f", weight=3, opacity=0.9,
                        dash_array="6", tooltip="Predicted route").add_to(pred)
        pred.add_to(m)

    # 4. raw GPS points (thinned if very long so the HTML stays small)
    stride = max(1, len(tr) // max_points)
    gps = folium.FeatureGroup(name=f"GPS points (every {stride})")
    for row in tr.iloc[::stride].itertuples():
        folium.CircleMarker([row.lat, row.lon], radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={row.t:.0f}s").add_to(gps)
    gps.add_to(m)

    folium.Marker([tr.lat.iloc[0], tr.lon.iloc[0]], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([tr.lat.iloc[-1], tr.lon.iloc[-1]], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
         padding: 10px 14px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
      <b>Trip {d['trip_id']}</b><br>
      <span style="color:#1565c0;">&#9679;</span> GPS observations ({len(tr)})<br>
      <span style="color:#9e9e9e;">&#9644;</span> Road network<br>
      <span style="color:#2e7d32;">&#9644;</span> Reference route<br>
      <span style="color:#d32f2f;">&#9476;</span> Predicted route
    </div>"""
    m.get_root().html.add_child(Element(legend))
    return m


if __name__ == "__main__":
    trip_dir = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/maps")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"trip_{trip_dir.name}.html"
    make_trip_map(trip_dir).save(f)
    print("Saved", f.resolve())
