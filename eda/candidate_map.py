"""Folium map that shows candidate generation and labelling for one trip.

For a window of GPS points it draws:
  - every candidate road segment found by pathsnap.candidates.build_candidates (orange)
  - the reference (correct) arc for each point from pathsnap.labels.align_reference_route (green)
  - GPS points coloured by outcome: green = correct arc is among the candidates,
    red = correct arc missing (a candidate miss), grey = point not usable for labelling

Run from the repo root:
    python3 eda/candidate_map.py data/student_test/00000052 results/maps --start 0 --count 150
"""
import argparse
import sys
from pathlib import Path

import folium
import pandas as pd
from branca.element import Element

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for "pathsnap"
from pathsnap.candidates import build_candidates, read_trip     # noqa: E402
from pathsnap.labels import align_reference_route               # noqa: E402

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=int, default=0, help="first GPS observation to show")
    ap.add_argument("--count", type=int, default=150, help="how many GPS observations to show")
    ap.add_argument("--radius", type=float, default=50)
    a = ap.parse_args()

    trip_dir = Path(a.trip_dir)
    track, nodes, arcs = read_trip(trip_dir)
    cand = build_candidates(trip_dir, radius=a.radius)
    ref = align_reference_route(trip_dir)

    obs = range(a.start, min(a.start + a.count, len(track)))
    cand = cand[cand.observation_id.isin(obs)]
    ref = ref[ref.observation_id.isin(obs)].set_index("observation_id")

    def line(arc_id):
        u, v = arcs.from_node[arc_id], arcs.to_node[arc_id]
        return [[nodes.latitude[u], nodes.longitude[u]], [nodes.latitude[v], nodes.longitude[v]]]

    win = track.iloc[list(obs)]
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(tiles=ESRI, attr="Tiles &copy; Esri", name="Esri streets").add_to(m)
    m.fit_bounds([[win.latitude.min(), win.longitude.min()], [win.latitude.max(), win.longitude.max()]])

    cand_ids = sorted(set(cand.arc_id))
    fg_c = folium.FeatureGroup(name=f"Candidate segments ({len(cand_ids)})")
    folium.PolyLine([line(i) for i in cand_ids], color="#ff9800", weight=3, opacity=0.8).add_to(fg_c)
    fg_c.add_to(m)

    ref_ids = sorted(set(ref.reference_arc_id))
    fg_r = folium.FeatureGroup(name="Reference arcs")
    folium.PolyLine([line(i) for i in ref_ids], color="#2e7d32", weight=7, opacity=0.6).add_to(fg_r)
    fg_r.add_to(m)

    per_obs = cand.groupby("observation_id").arc_id.apply(set)
    fg_p = folium.FeatureGroup(name="GPS points")
    hit = miss = unusable = 0
    for i in obs:
        r = ref.loc[i]
        n_c = len(per_obs.get(i, ()))
        if r.label_usable != 1:
            colour, status = "#757575", "not usable for labelling"
            unusable += 1
        elif r.reference_arc_id in per_obs.get(i, ()):
            colour, status = "#2e7d32", "correct arc is a candidate"
            hit += 1
        else:
            colour, status = "#d32f2f", "CANDIDATE MISS"
            miss += 1
        folium.CircleMarker(
            [track.latitude[i], track.longitude[i]], radius=4, color=colour, fill=True, fill_opacity=0.9,
            weight=1, tooltip=(f"obs {i} | {n_c} candidates | ref arc {int(r.reference_arc_id)} | "
                               f"{r.alignment_distance_m:.1f} m from route | {status}")).add_to(fg_p)
    fg_p.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    usable = hit + miss
    rate = miss / usable if usable else float("nan")
    avg_c = cand.groupby("observation_id").size().mean()
    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
         padding: 10px 14px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
      <b>Candidate generation, trip {trip_dir.name}</b><br>
      Observations {obs.start} to {obs.stop - 1}, radius {a.radius:.0f} m<br>
      Average candidates per point: {avg_c:.1f}<br>
      Candidate miss rate: {rate:.3f} ({miss} of {usable} usable points)<br>
      <span style="color:#ff9800;">&#9644;</span> Candidate segments<br>
      <span style="color:#2e7d32;">&#9644;</span> Reference arc<br>
      <span style="color:#2e7d32;">&#9679;</span> Correct arc found
      <span style="color:#d32f2f;">&#9679;</span> Missed
      <span style="color:#757575;">&#9679;</span> Not usable ({unusable})
    </div>"""
    m.get_root().html.add_child(Element(legend))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"candidates_{trip_dir.name}_{obs.start}_{obs.stop - 1}.html"
    m.save(f)
    print(f"Saved {f}  | avg candidates {avg_c:.1f} | miss rate {rate:.3f}")


if __name__ == "__main__":
    main()
