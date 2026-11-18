
from pathlib import Path
import logging
import os
import re
import time

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from pathsnap.candidates import (
    FEATURE_COLUMNS,
    build_candidates,
    read_trip,
    project_coordinates,
)
from pathsnap.hmm import viterbi_decode
from pathsnap.labels import align_reference_route


ROOT = Path(__file__).resolve().parents[1]

# Override this with an environment variable when using evaluator_test.
EVAL_DIR = Path(os.getenv("EVAL_DIR", str(ROOT / "data" / "student_test")))
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(ROOT / "models" / "mlp.joblib"))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pathsnap-api")

app = FastAPI(title="PathSnap API", version="1.0.0")
model = None


class PredictionRequest(BaseModel):
    trip_id: str


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info("MLP model loaded from %s", MODEL_PATH)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        logger.info(
            "%s %s -> %s (%.3fs)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            "%s %s failed after %.3fs",
            request.method,
            request.url.path,
            elapsed,
        )
        raise


def get_trip_directory(trip_id: str) -> Path:
    # Reject path traversal and unexpected identifiers.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", trip_id):
        raise HTTPException(status_code=400, detail="Invalid trip ID")

    trip_dir = EVAL_DIR / trip_id

    if not trip_dir.is_dir():
        raise HTTPException(status_code=404, detail="Trip not found")

    required_files = [
        f"{trip_id}.track",
        f"{trip_id}.nodes",
        f"{trip_id}.arcs",
    ]

    missing = [
        filename for filename in required_files
        if not (trip_dir / filename).is_file()
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Trip is missing required files: {missing}",
        )

    return trip_dir


@app.get("/")
def health():
    return {
        "status": "online",
        "service": "PathSnap API",
        "model_loaded": model is not None,
    }


@app.get("/trips")
def list_trips():
    if not EVAL_DIR.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation directory does not exist: {EVAL_DIR}",
        )

    trips = []

    for folder in sorted(EVAL_DIR.iterdir()):
        if not folder.is_dir():
            continue

        trip_id = folder.name

        if not re.fullmatch(r"[A-Za-z0-9_-]+", trip_id):
            continue

        required = [
            folder / f"{trip_id}.track",
            folder / f"{trip_id}.nodes",
            folder / f"{trip_id}.arcs",
        ]

        if all(path.is_file() for path in required):
            trips.append(trip_id)

    return {"trips": trips, "count": len(trips)}


def build_route_coordinates(predictions, arcs, nodes):
    """Convert predicted arc IDs into a [longitude, latitude] route."""

    coordinates = []

    for arc_id in predictions["predicted_arc_id"].tolist():
        arc_id = int(arc_id)

        if arc_id < 0 or arc_id >= len(arcs):
            continue

        start_node = int(arcs.iloc[arc_id, 0])
        end_node = int(arcs.iloc[arc_id, 1])

        if not (
            0 <= start_node < len(nodes)
            and 0 <= end_node < len(nodes)
        ):
            continue

        start = [
            float(nodes.iloc[start_node, 0]),
            float(nodes.iloc[start_node, 1]),
        ]
        end = [
            float(nodes.iloc[end_node, 0]),
            float(nodes.iloc[end_node, 1]),
        ]

        if not coordinates or coordinates[-1] != start:
            coordinates.append(start)

        if coordinates[-1] != end:
            coordinates.append(end)

    return coordinates

def build_road_network_coordinates(track, nodes, arcs, margin=0.005):
    """
    Return road segments within a small bounding box around the GPS trace.
    Coordinates are returned as [longitude, latitude].
    """
    gps_lon = track.iloc[:, 0].astype(float)
    gps_lat = track.iloc[:, 1].astype(float)

    min_lon = gps_lon.min() - margin
    max_lon = gps_lon.max() + margin
    min_lat = gps_lat.min() - margin
    max_lat = gps_lat.max() + margin

    road_segments = []

    for _, arc in arcs.iterrows():
        start_id = int(arc.iloc[0])
        end_id = int(arc.iloc[1])

        if not (0 <= start_id < len(nodes) and 0 <= end_id < len(nodes)):
            continue

        start = nodes.iloc[start_id, :2].astype(float).tolist()
        end = nodes.iloc[end_id, :2].astype(float).tolist()

        # Keep roads intersecting the GPS bounding box.
        if (
            min_lon <= start[0] <= max_lon
            and min_lat <= start[1] <= max_lat
        ) or (
            min_lon <= end[0] <= max_lon
            and min_lat <= end[1] <= max_lat
        ):
            road_segments.append([start, end])

    return road_segments

@app.post("/predict")
def predict_route(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    trip_dir = get_trip_directory(request.trip_id)
    started = time.perf_counter()

    try:
        # Read raw trip data.
        track, nodes, arcs = read_trip(trip_dir)
        # Raw GPS coordinates: [longitude, latitude]
        gps_coordinates = (
            track.iloc[:, :2]
            .astype(float)
            .values
            .tolist()
        )

        # Generate candidates without loading reference labels.
        candidates = build_candidates(trip_dir)

        if candidates.empty:
            raise HTTPException(
                status_code=422,
                detail="No road candidates generated for this trip",
            )

        # Predict candidate probabilities.
        X = candidates[FEATURE_COLUMNS].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

        positive_class_index = list(model.classes_).index(1)
        candidates = candidates.copy()
        candidates["candidate_probability"] = model.predict_proba(X)[
            :, positive_class_index
        ]

        # Project nodes for Viterbi transition calculations.
        _, node_xy, _ = project_coordinates(track, nodes)

        predictions = viterbi_decode(
            candidates,
            arcs,
            node_xy,
        )

        route_coordinates = build_route_coordinates(
            predictions, arcs, nodes
        )

        road_network = build_road_network_coordinates(track, nodes, arcs)

        # Ground truth is used ONLY on the server for evaluation.
        metrics = {
            "observations": int(track.shape[0]),
            "candidate_count": int(len(candidates)),
            "evaluated_observations": 0,
            "arc_accuracy": None,
            "candidate_miss_rate": None,
        }

        route_file = trip_dir / f"{request.trip_id}.route"

        if route_file.exists():
            reference = align_reference_route(trip_dir)

            evaluated = reference[
                reference["label_usable"] == 1
            ].merge(
                predictions,
                on="observation_id",
                how="inner",
            )

            if not evaluated.empty:
                metrics["evaluated_observations"] = int(len(evaluated))
                metrics["arc_accuracy"] = float(
                    (
                        evaluated["predicted_arc_id"]
                        == evaluated["reference_arc_id"]
                    ).mean()
                )

            # Candidate miss rate: usable observations whose true arc
            # was absent from the generated candidate set.
            usable_reference = reference[
                reference["label_usable"] == 1
            ]

            candidate_matches = candidates.merge(
                usable_reference[["observation_id", "reference_arc_id"]],
                on="observation_id",
                how="inner",
            )

            if not usable_reference.empty:
                found = (
                    candidate_matches["arc_id"]
                    == candidate_matches["reference_arc_id"]
                ).groupby(candidate_matches["observation_id"]).any()

                missed = len(usable_reference) - int(found.sum())

                metrics["candidate_miss_rate"] = float(
                    missed / len(usable_reference)
                )

        metrics["prediction_time_seconds"] = round(
            time.perf_counter() - started, 4
        )

        logger.info(
            "Prediction completed for trip %s: %d predicted observations",
            request.trip_id,
            len(predictions),
        )

        # Never return reference labels, reference route, or true arc IDs.
        return {
            "trip_id": request.trip_id,
            "gps_coordinates": gps_coordinates,
            "predicted_coordinates": route_coordinates,
            "predicted_observations": int(len(predictions)),
            "road_network": road_network,
            "metrics": metrics
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for trip %s", request.trip_id)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
