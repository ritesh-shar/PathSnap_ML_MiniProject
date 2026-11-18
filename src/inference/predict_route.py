"""End-to-end PathSnap route prediction."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates import (
    build_candidates,
    read_trip,
    project_coordinates,
)
from pathsnap.hmm import viterbi_decode


def predict_route(trip_directory, model_path=None, transition_mode="straight"):
    """
    Predict one road arc per GPS observation.

    If model_path is omitted, use distance-based scores
    for smoke testing only.
    """
    trip_directory = Path(trip_directory)

    # 1. Load raw trip and projected road network.
    track, nodes, arcs = read_trip(trip_directory)
    _, node_xy, _ = project_coordinates(track, nodes)

    # 2. Generate candidate roads and features.
    candidates = build_candidates(trip_directory)

    # 3. Obtain emission probabilities.
    if model_path:
        from src.inference.model_scores import score_candidates

        scored = score_candidates(candidates, model_path)
        score_source = "trained_classifier"
    else:
        scored = candidates.copy()
        scored["candidate_probability"] = np.exp(
            -scored["distance_m"].to_numpy(dtype=float) / 20.0
        )
        score_source = "distance_smoke_test"

    # 4. Independent classifier/candidate baseline.
    baseline = (
        scored.sort_values(
            "candidate_probability",
            ascending=False,
            kind="stable",
        )
        .drop_duplicates("observation_id")
        .sort_values("observation_id")
        [["observation_id", "arc_id"]]
        .rename(columns={"arc_id": "baseline_arc_id"})
    )

    # 5. Decode the complete sequence with HMM.
    decoded = viterbi_decode(
        scored,
        arcs,
        node_xy,
        probability_col="candidate_probability",
        transition_mode=transition_mode,
    )

    # 6. Combine results, preserving GPS order.
    result = (
        decoded.merge(
            baseline,
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("observation_id")
        .reset_index(drop=True)
    )

    result.insert(0, "trip_id", trip_directory.name)
    result["score_source"] = score_source

    if len(result) != len(track):
        raise ValueError(
            f"Expected {len(track)} predictions, "
            f"received {len(result)}"
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--transition-mode",
        choices=["straight", "network"],
        default="straight",
    )
    parser.add_argument(
        "--out",
        default="results/inference/route_predictions.csv",
    )
    args = parser.parse_args()

    result = predict_route(args.trip, args.model, args.transition_mode)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    changed = (
        result["predicted_arc_id"]
        != result["baseline_arc_id"]
    ).sum()

    print("Trip:", result["trip_id"].iloc[0])
    print("GPS observations:", len(result))
    print("Different from baseline:", int(changed))
    print("Scoring source:", result["score_source"].iloc[0])
    print("Saved:", output)


if __name__ == "__main__":
    main()
