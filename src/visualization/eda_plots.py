
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

Path("results/figures").mkdir(parents=True, exist_ok=True)

# Load training EDA summary
eda = pd.read_csv("results/train_eda.csv")

# Plot 1: GPS observations per trip
plt.figure(figsize=(14, 5))
plt.bar(eda["trip_id"].astype(str), eda["gps_points"])
plt.title("GPS Observations per Training Trip")
plt.xlabel("Trip ID")
plt.ylabel("Number of GPS Observations")
plt.xticks([])
plt.tight_layout()
plt.savefig("results/figures/gps_counts.png", dpi=200)
plt.close()

# Plot 2: Distribution of GPS observations
plt.figure(figsize=(8, 5))
plt.hist(eda["gps_points"], bins=20, edgecolor="black")
plt.title("Distribution of Training Trip Lengths")
plt.xlabel("GPS Observations per Trip")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig("results/figures/trip_length_distribution.png", dpi=200)
plt.close()

# Plot 3: Actual GPS trajectory for a training trip
trip = "00000001"

track = pd.read_csv(
    f"train/{trip}/{trip}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

plt.figure(figsize=(8, 7))
plt.plot(
    track["longitude"],
    track["latitude"],
    linewidth=1,
    alpha=0.7
)
plt.scatter(
    track["longitude"].iloc[0],
    track["latitude"].iloc[0],
    marker="o",
    s=65,
    label="Start"
)
plt.scatter(
    track["longitude"].iloc[-1],
    track["latitude"].iloc[-1],
    marker="x",
    s=65,
    label="End"
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Raw GPS Trajectory — Trip {trip}")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/gps_trajectory.png", dpi=200)
plt.close()

print("All three EDA plots generated successfully!")
