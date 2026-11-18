"""Folium map comparing classifier-only predictions with HMM/Viterbi predictions for one trip.

Input: a predictions CSV with one row per GPS observation and columns
    observation_id, <classifier column>, <hmm column>
(by default baseline_arc_id and predicted_arc_id, the format Member 2's decoder writes).

Run from the repo root:
    python3 eda/route_compare_map.py data/student_test/00000052 results/predictions/00000052.csv results/maps

It also prints and saves the pointwise accuracy of both methods against the reference labels
(usable points only) to results/maps/route_compare_<trip>.csv.
"""
import argparse
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from branca.element import Element

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.candidates import read_trip          # noqa: E402
from pathsnap.labels import align_reference_route  # noqa: E402

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"


def continuity(arc_seq, arcs):
    """Share of consecutive changes of arc where the two arcs actually connect in the road network."""
    seq = [a for i, a in enumerate(arc_seq) if i == 0 or a != arc_seq[i - 1]]
    if len(seq) < 2:
        return float("nan")
    ends, starts = arcs.to_node.values, arcs.from_node.values
    ok = sum(ends[a] == starts[b] or starts[a] == starts[b] or ends[a] == ends[b] or starts[a] == ends[b]
             for a, b in zip(seq[:-1], seq[1:]))
    return ok / (len(seq) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_dir")
    ap.add_argument("pred_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--clf-col", default="baseline_arc_id")
    ap.add_argument("--hmm-col", default="predicted_arc_id")
    a = ap.parse_args()

    trip_dir = Path(a.trip_dir)
    tid = trip_dir.name
    track, nodes, arcs = read_trip(trip_dir)
    pred = pd.read_csv(a.pred_csv).sort_values("observation_id")
    ref = align_reference_route(trip_dir)
    route = pd.read_csv(trip_dir / f"{tid}.route", header=None, names=["arc_id"]).arc_id.values

    # pointwise accuracy on usable points
    m_ = pred.merge(ref, on="observation_id", how="inner")
    use = m_[m_.label_usable == 1]
    stats = {
        "trip_id": tid,
        "usable_points": len(use),
        "classifier_top1": (use[a.clf_col] == use.reference_arc_id).mean(),
        "hmm_top1": (use[a.hmm_col] == use.reference_arc_id).mean(),
        "classifier_continuity": continuity(pred[a.clf_col].tolist(), arcs),
        "hmm_continuity": continuity(pred[a.hmm_col].tolist(), arcs),
    }

    def lines(arc_ids):
        u, v = arcs.from_node.values[arc_ids], arcs.to_node.values[arc_ids]
        return [[[nodes.latitude[p], nodes.longitude[p]], [nodes.latitude[q], nodes.longitude[q]]]
                for p, q in zip(u, v)]

    def dedup(seq):
        seq = np.asarray(seq)
        return seq[np.r_[True, seq[1:] != seq[:-1]]]

    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(tiles=ESRI, attr="Tiles &copy; Esri", name="Esri streets").add_to(m)
    m.fit_bounds([[track.latitude.min(), track.longitude.min()], [track.latitude.max(), track.longitude.max()]])

    layers = [
        ("Reference route", route, "#2e7d32", 8, 0.5, None),
        ("Classifier only", dedup(pred[a.clf_col]), "#ff9800", 4, 0.9, None),
        ("Classifier + HMM", dedup(pred[a.hmm_col]), "#d32f2f", 3, 0.9, "6"),
    ]
    for name, ids, colour, w, op, dash in layers:
        fg = folium.FeatureGroup(name=name)
        folium.PolyLine(lines(ids), color=colour, weight=w, opacity=op, dash_array=dash, tooltip=name).add_to(fg)
        fg.add_to(m)

    stride = max(1, len(track) // 1500)
    fg = folium.FeatureGroup(name="GPS points")
    for r in track.iloc[::stride].itertuples():
        folium.CircleMarker([r.latitude, r.longitude], radius=2, color="#1565c0", fill=True,
                            fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
         padding: 10px 14px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
      <b>Trip {tid}: classifier vs HMM</b><br>
      <span style="color:#1565c0;">&#9679;</span> GPS points<br>
      <span style="color:#2e7d32;">&#9644;</span> Reference route<br>
      <span style="color:#ff9800;">&#9644;</span> Classifier only: Top-1 {stats['classifier_top1']:.3f},
        continuity {stats['classifier_continuity']:.3f}<br>
      <span style="color:#d32f2f;">&#9476;</span> Classifier + HMM: Top-1 {stats['hmm_top1']:.3f},
        continuity {stats['hmm_continuity']:.3f}<br>
      ({stats['usable_points']} usable points)
    </div>"""
    m.get_root().html.add_child(Element(legend))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / f"route_compare_{tid}.html")
    pd.DataFrame([stats]).to_csv(out / f"route_compare_{tid}.csv", index=False)
    print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in stats.items()})
    print("Saved", out / f"route_compare_{tid}.html")


if __name__ == "__main__":
    main()
