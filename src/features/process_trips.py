
from pathlib import Path
import argparse
from pathsnap.candidates import build_candidates

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

success = 0

for trip in folders:
    try:
        features = build_candidates(
            trip,
            radius=args.radius,
            max_candidates=12
        )

        output_file = (
            output_dir / f"{trip.name}_candidates.csv"
        )

        features.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

        success += 1

    except Exception as error:
        print(f"ERROR processing {trip.name}: {error}")
        raise

print(f"\nSuccessfully processed {success} trips.")
