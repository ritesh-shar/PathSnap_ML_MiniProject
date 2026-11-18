"""Shared helpers for loading one PathSnap trip and converting coordinates to metres."""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

# Trips are spread across the world, so each trip is projected into its own UTM zone
# (chosen from the trip's mean longitude and hemisphere). Within one zone distances are
# accurate to well under 1%, which raw degrees are not (1 degree of longitude shrinks with latitude).
_cache = {}


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def transformer_for(lon, lat):
    epsg = utm_epsg(float(np.mean(lon)), float(np.mean(lat)))
    if epsg not in _cache:
        _cache[epsg] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return _cache[epsg]


def list_trips(split_dir):
    """Return the sorted trip folders inside a split folder (train or student_test)."""
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir())


def load_trip(trip_dir):
    """Load the four raw files of one trip. Missing files come back as None."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name

    def read(ext, names, dtype):
        f = trip_dir / f"{tid}.{ext}"
        if not f.exists():
            return None
        return pd.read_csv(f, sep=r"\s+", header=None, names=names, dtype=dtype)

    return {
        "trip_id": tid,
        "track": read("track", ["lon", "lat", "t"], float),
        "nodes": read("nodes", ["lon", "lat"], float),
        "arcs": read("arcs", ["u", "v"], "int64"),
        "route": read("route", ["arc_id"], "int64"),
    }


def to_xy(lon, lat, ref_lon=None, ref_lat=None):
    """Convert lon/lat arrays to UTM metres (x, y).

    Pass the trip's GPS lon/lat as ref_lon/ref_lat when projecting nodes, so that the
    GPS points and the road network of one trip always use the same zone.
    """
    ref_lon = lon if ref_lon is None else ref_lon
    ref_lat = lat if ref_lat is None else ref_lat
    x, y = transformer_for(ref_lon, ref_lat).transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def point_to_segments_dist(px, py, ax, ay, bx, by, chunk=2000):
    """Minimum distance in metres from each point to a set of line segments A-B."""
    dx, dy = bx - ax, by - ay
    len2 = np.where(dx * dx + dy * dy == 0, 1e-9, dx * dx + dy * dy)
    out = np.empty(len(px))
    for s in range(0, len(px), chunk):
        X, Y = px[s:s + chunk, None], py[s:s + chunk, None]
        tt = np.clip(((X - ax) * dx + (Y - ay) * dy) / len2, 0, 1)
        d = np.hypot(X - (ax + tt * dx), Y - (ay + tt * dy))
        out[s:s + chunk] = d.min(axis=1)
    return out
