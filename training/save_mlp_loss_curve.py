import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from training.harness import MODEL_FACTORIES, DEFAULT_FEATURES

DATA_PATH = ROOT / "results" / "train_features.csv"
OUTPUT_DIR = ROOT / "results"

SEED = 42

print("Loading training data...")

df = pd.read_csv(DATA_PATH)

# Keep only observations with usable labels
df = df[df["label_usable"] == 1].copy()

X = df[DEFAULT_FEATURES].astype(float)
y = df["label"].astype(int)

print(f"Training rows: {len(df):,}")
print(f"Features: {DEFAULT_FEATURES}")

# Same MLP configuration used in the original experiments
model = MODEL_FACTORIES["mlp"](SEED)

print("\nTraining MLP and recording loss history...")

model.fit(X, y)

# Extract the actual fitted MLP from the pipeline
mlp = model.named_steps["mlpclassifier"]

# Actual epoch-wise training loss
history = pd.DataFrame({
    "epoch": range(1, len(mlp.loss_curve_) + 1),
    "training_loss": mlp.loss_curve_
})

# Validation accuracy is available, but it is NOT validation loss.
if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
    history["validation_accuracy"] = pd.Series(
        mlp.validation_scores_
    )

# Save numerical history
csv_path = OUTPUT_DIR / "mlp_loss_history.csv"
history.to_csv(csv_path, index=False)

# Save training loss plot
plt.figure(figsize=(9, 5))
plt.plot(
    history["epoch"],
    history["training_loss"],
    marker="o",
    label="Training loss"
)

plt.title("MLP Training Loss Curve")
plt.xlabel("Iteration (epoch)")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

png_path = OUTPUT_DIR / "mlp_loss_curve.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

# Save validation accuracy separately, if available
if "validation_accuracy" in history.columns:
    plt.figure(figsize=(9, 5))
    plt.plot(
        history["epoch"],
        history["validation_accuracy"],
        marker="o",
        label="Validation accuracy"
    )
    plt.title("MLP Validation Accuracy")
    plt.xlabel("Iteration (epoch)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "mlp_validation_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\nSaved:")
print(csv_path)
print(png_path)

if "validation_accuracy" in history.columns:
    print(OUTPUT_DIR / "mlp_validation_accuracy.png")

print(f"\nIterations completed: {len(mlp.loss_curve_)}")
print(f"Final training loss: {mlp.loss_curve_[-1]:.6f}")
