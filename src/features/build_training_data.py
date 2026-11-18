
from pathlib import Path
import argparse
import pandas as pd

from pathsnap.candidates import build_candidates
from pathsnap.labels import align_reference_route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--radius", type=float, default=50)
    args = parser.parse_args()

    folders = sorted(
        p for p in Path(args.partition).iterdir()
        if p.is_dir()
    )

    if args.limit > 0:
        folders = folders[:args.limit]

    output_dir = Path("results/features") / args.partition
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_dir = Path("results/coverage")
    coverage_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for folder in folders:
        candidates = build_candidates(
            folder,
            radius=args.radius,
            max_candidates=12
        )

        reference = align_reference_route(folder)

        merged = candidates.merge(
            reference,
            on=["trip_id", "observation_id"],
            how="left",
            validate="many_to_one"
        )

        merged["label"] = (
            merged["arc_id"] == merged["reference_arc_id"]
        ).astype(int)

        usable = merged[merged["label_usable"] == 1]

        positive_per_obs = usable.groupby(
            "observation_id"
        )["label"].max()

        candidate_miss_rate = (
            1 - positive_per_obs.mean()
            if len(positive_per_obs)
            else float("nan")
        )

        output = output_dir / f"{folder.name}_features.csv"
        merged.to_csv(output, index=False)

        summaries.append({
            "trip_id": folder.name,
            "radius_m": args.radius,
            "total_observations": len(reference),
            "usable_observations": len(positive_per_obs),
            "provisional_candidate_miss_rate": candidate_miss_rate
        })

        print(
            f"{folder.name}: "
            f"usable={len(positive_per_obs)}, "
            f"miss_rate={candidate_miss_rate:.3f}, "
            f"saved={output}"
        )

    summary_file = (
        coverage_dir / f"{args.partition}_coverage.csv"
    )

    pd.DataFrame(summaries).to_csv(
        summary_file,
        index=False
    )

    print(f"\nProcessed {len(summaries)} trips")
    print(f"Coverage results saved: {summary_file}")


if __name__ == "__main__":
    main()
