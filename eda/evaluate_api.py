"""End-to-end evaluation of the deployed PathSnap system through its API.

Calls the running FastAPI server (Laptop 1) for every trip it lists, exactly as the UI does,
and records the metrics the server computes (accuracy is computed on the server side, the
reference route never leaves the API).

Run from the repo root on Laptop 2 while the API is running on Laptop 1:
    python eda/evaluate_api.py http://10.1.4.169:8000 results/api_evaluation

Outputs:
    api_evaluation.csv          one row per trip
    api_summary.csv             mean / median / min / max over all trips
    api_accuracy_per_trip.png   arc accuracy per trip with the overall mean
    api_time_vs_points.png      prediction time against number of GPS points
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests


def main(api_url, out_dir):
    api_url = api_url.rstrip("/")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trips = requests.get(f"{api_url}/trips", timeout=15).json()["trips"]
    print(f"API lists {len(trips)} trips: {trips}")

    rows = []
    for tid in trips:
        t0 = time.time()
        try:
            r = requests.post(f"{api_url}/predict", json={"trip_id": tid}, timeout=300)
            r.raise_for_status()
            res = r.json()
            m = res["metrics"]
            rows.append({
                "trip_id": tid,
                "status": "ok",
                "gps_points": m.get("observations"),
                "evaluated_points": m.get("evaluated_observations"),
                "candidate_rows": m.get("candidate_count"),
                "candidates_per_point": (m.get("candidate_count") or 0) / max(1, m.get("observations") or 1),
                "arc_accuracy": m.get("arc_accuracy"),
                "candidate_miss_rate": m.get("candidate_miss_rate"),
                "server_time_s": m.get("prediction_time_seconds"),
                "round_trip_time_s": round(time.time() - t0, 3),
            })
        except Exception as exc:  # keep going so one bad trip does not stop the evaluation
            rows.append({"trip_id": tid, "status": f"error: {exc}"})
        r_ = rows[-1]
        print(f"{tid}: {r_['status']}  accuracy={r_.get('arc_accuracy')}  "
              f"miss={r_.get('candidate_miss_rate')}  time={r_.get('server_time_s')}s")

    df = pd.DataFrame(rows)
    df["trip_id"] = df["trip_id"].astype(str)
    df.to_csv(out / "api_evaluation.csv", index=False)
    ok = df[df.status == "ok"].copy()
    if ok.empty:
        print("No successful predictions.")
        return

    cols = ["arc_accuracy", "candidate_miss_rate", "server_time_s", "round_trip_time_s",
            "gps_points", "candidates_per_point"]
    summ = ok[cols].astype(float).agg(["mean", "median", "min", "max"]).T
    # weighted accuracy: every evaluated GPS point counts equally, not every trip
    w = (ok.arc_accuracy * ok.evaluated_points).sum() / ok.evaluated_points.sum()
    summ.loc["arc_accuracy_point_weighted", "mean"] = w
    summ.to_csv(out / "api_summary.csv")
    print("\n", summ.round(4).to_string())

    ok = ok.sort_values("arc_accuracy")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ok.trip_id, ok.arc_accuracy * 100, color="#4c72b0")
    ax.axhline(ok.arc_accuracy.mean() * 100, color="#d62728", ls="--",
               label=f"mean over trips {ok.arc_accuracy.mean() * 100:.2f}%")
    for i, v in enumerate(ok.arc_accuracy * 100):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="Trip", ylabel="Arc accuracy (%)", ylim=(0, 105),
           title="Deployed system (API) arc accuracy per trip")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(out / "api_accuracy_per_trip.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ok.gps_points, ok.server_time_s, color="#4c72b0")
    for _, r in ok.iterrows():
        ax.annotate(r.trip_id[-3:], (r.gps_points, r.server_time_s), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="GPS points in trip", ylabel="Server prediction time (s)",
           title="Prediction time vs trip length")
    fig.tight_layout()
    fig.savefig(out / "api_time_vs_points.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved results to {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results/api_evaluation")
