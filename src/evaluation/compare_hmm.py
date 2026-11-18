from pathlib import Path
from time import perf_counter

import pandas as pd

from pathsnap.candidates import read_trip, project_coordinates
from pathsnap.hmm import viterbi_decode
from src.inference.model_scores import score_candidates


FEATURE_DIR = Path("results/features/student_test")
TRIP_DIR = Path("student_test")
MODEL_PATH = "models/best_model.joblib"
OUTPUT = Path("results/hmm_comparison.csv")


def evaluate_trip(feature_file):
    trip_id = feature_file.name.replace("_features.csv", "")

    print(f"\nEvaluating trip {trip_id}", flush=True)

    features = pd.read_csv(feature_file, dtype={"trip_id": str})

    track, nodes, arcs = read_trip(TRIP_DIR / trip_id)
    _, node_xy, _ = project_coordinates(track, nodes)

    # The model receives only allowed numerical features.
    start = perf_counter()
    scored = score_candidates(features, MODEL_PATH)
    model_seconds = perf_counter() - start

    baseline = (
        scored.sort_values(
            "candidate_probability",
            ascending=False,
            kind="stable"
        )
        .drop_duplicates("observation_id")
        [["observation_id", "arc_id"]]
        .rename(columns={"arc_id": "baseline_arc_id"})
    )

    start = perf_counter()
    decoded = viterbi_decode(
        scored,
        arcs,
        node_xy,
        probability_col="candidate_probability",
        transition_mode="network",
    )
    hmm_seconds = perf_counter() - start

    predictions = baseline.merge(
        decoded,
        on="observation_id",
        validate="one_to_one"
    )

    # Reference labels are used ONLY for evaluation.
    usable = features[features["label_usable"] == 1]

    reference = usable[
        ["observation_id", "reference_arc_id"]
    ].drop_duplicates("observation_id")

    evaluated = predictions.merge(
        reference,
        on="observation_id",
        validate="one_to_one",
        how="inner"
    ).dropna(subset=["reference_arc_id"])

    ml_correct = int((
        evaluated["baseline_arc_id"]
        == evaluated["reference_arc_id"]
    ).sum())

    hmm_correct = int((
        evaluated["predicted_arc_id"]
        == evaluated["reference_arc_id"]
    ).sum())

    n = len(evaluated)
    if n == 0:
        raise ValueError(f"No usable labels for {trip_id}")

    changed = int((
        predictions["baseline_arc_id"]
        != predictions["predicted_arc_id"]
    ).sum())

    if len(predictions) != len(track):
        raise ValueError(
            f"Incomplete predictions for {trip_id}"
        )

    result = {
        "trip_id": trip_id,
        "gps_points": len(track),
        "usable_points": n,
        "ml_correct": ml_correct,
        "hmm_correct": hmm_correct,
        "ml_agreement": ml_correct / n,
        "hmm_agreement": hmm_correct / n,
        "improvement_pp": (
            (hmm_correct - ml_correct) / n * 100
        ),
        "changed_predictions": changed,
        "model_seconds": model_seconds,
        "hmm_seconds": hmm_seconds,
    }

    print(
        f"ML: {result['ml_agreement']:.4f} | "
        f"HMM: {result['hmm_agreement']:.4f} | "
        f"Change: {result['improvement_pp']:+.2f} pp",
        flush=True
    )

    return result


def main():
    files = sorted(FEATURE_DIR.glob("*_features.csv"))
    if not files:
        raise FileNotFoundError("No student-test feature files")

    rows = []

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for file in files:
        rows.append(evaluate_trip(file))

        # Save incrementally so completed results survive interruption.
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)

    results = pd.DataFrame(rows)

    n = results["usable_points"].sum()
    ml = results["ml_correct"].sum() / n
    hmm = results["hmm_correct"].sum() / n

    print("\nFINAL SUMMARY")
    print("Trips evaluated:", len(results))
    print("Usable reference observations:", n)
    print(f"Overall ML agreement: {ml:.4f}")
    print(f"Overall HMM agreement: {hmm:.4f}")
    print(f"Overall difference: {(hmm-ml)*100:+.2f} pp")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
