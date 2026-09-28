from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# YOLO11 Currency Detection Prediction
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "yolo11"
MODEL_PATH = PROJECT_ROOT / "runs" / "currency_yolo11n-3" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo11_predictions"

# ---------------------------------------------------------
# Check model
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    print("=" * 70)
    print("YOLO11 MODEL NOT FOUND")
    print("=" * 70)
    print()
    print(f"Expected model:")
    print(MODEL_PATH)
    print()
    print("We have NOT trained YOLO11 yet.")
    print("So prediction cannot be run yet.")
    print()
    print("Next step: train YOLO11 first.")
    raise SystemExit(1)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("=" * 70)
print("Indian Currency YOLO11 Prediction")
print("=" * 70)
print()
print(f"Model  : {MODEL_PATH}")
print(f"Output : {OUTPUT_DIR}")
print()

model = YOLO(str(MODEL_PATH))

# ---------------------------------------------------------
# Predict on TEST images
# ---------------------------------------------------------

test_images = DATASET_DIR / "images" / "test"

if not test_images.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{test_images}"
    )

print("Running YOLO11 on test images...")
print()

results = model.predict(
    source=str(test_images),
    conf=0.25,
    iou=0.45,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(OUTPUT_DIR.parent),
    name=OUTPUT_DIR.name,
    exist_ok=True,
    verbose=True,
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)
print()
print(f"Results saved to:")
print(OUTPUT_DIR)
print()
print("Open the prediction images and inspect the boxes.")
