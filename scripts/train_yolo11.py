from pathlib import Path
from ultralytics import YOLO
import torch

# ============================================================
# Indian Currency YOLO11 Training
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = PROJECT_ROOT / "dataset" / "yolo11" / "data.yaml"
MODEL = PROJECT_ROOT / "yolo11n.pt"

RUNS_DIR = PROJECT_ROOT / "runs"

print("=" * 70)
print("Indian Currency YOLO11 Training")
print("=" * 70)
print()

# ------------------------------------------------------------
# Verify dataset
# ------------------------------------------------------------

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset configuration not found:\n{DATA_YAML}"
    )

if not MODEL.exists():
    raise FileNotFoundError(
        f"YOLO11 pretrained model not found:\n{MODEL}"
    )

print(f"Dataset : {DATA_YAML}")
print(f"Model   : {MODEL}")
print()

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

# Your machine has Intel UHD Graphics and no NVIDIA CUDA GPU.
# Therefore PyTorch/Ultralytics will use CPU.
device = "cpu"

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Training device : {device}")
print()

# ------------------------------------------------------------
# Load pretrained YOLO11n
# ------------------------------------------------------------

model = YOLO(str(MODEL))

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

results = model.train(
    data=str(DATA_YAML),

    # YOLO11 nano: appropriate for CPU experimentation
    epochs=50,

    # Keep image size moderate for CPU training
    imgsz=640,

    # CPU training
    device=device,

    # Small batch because this is CPU-only
    batch=8,

    # Number of dataloader workers
    workers=2,

    # Reproducibility
    seed=42,

    # Standard training controls
    patience=15,

    # Save checkpoints
    save=True,

    # Project organization
    project=str(RUNS_DIR),
    name="currency_yolo11n",

    # Validation during training
    val=True,

    # Cache disabled to avoid unnecessary RAM/disk pressure
    cache=False,

    # Use pretrained weights
    pretrained=True,

    # Keep plots and metrics
    plots=True,

    verbose=True,
)

print()
print("=" * 70)
print("YOLO11 TRAINING COMPLETE")
print("=" * 70)
print()

print(f"Training output:")
print(RUNS_DIR / "currency_yolo11n")

print()
print("Best model should be:")
print(RUNS_DIR / "currency_yolo11n" / "weights" / "best.pt")