from pathlib import Path
from PIL import Image

# ============================================================
# Create MobileNetV2 Classification Dataset
# From existing YOLO11 dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_DIR = PROJECT_ROOT / "dataset" / "yolo11"
MOBILE_DIR = PROJECT_ROOT / "dataset" / "mobilenet"

CLASS_NAMES = {
    0: "10",
    1: "20",
    2: "50",
    3: "100",
    4: "200",
    5: "500",
    6: "2000",
}

SPLITS = ["train", "val", "test"]

print("=" * 70)
print("Creating MobileNetV2 Classification Dataset")
print("=" * 70)
print()

# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------

for split in SPLITS:
    for class_name in CLASS_NAMES.values():
        (MOBILE_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

# ------------------------------------------------------------
# Process images
# ------------------------------------------------------------

total = 0
counts = {
    split: {class_name: 0 for class_name in CLASS_NAMES.values()}
    for split in SPLITS
}

for split in SPLITS:

    image_dir = YOLO_DIR / "images" / split
    label_dir = YOLO_DIR / "labels" / split

    print(f"Processing {split.upper()}...")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    for index, image_path in enumerate(images, start=1):

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"WARNING: Missing label: {image_path.name}")
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if len(lines) != 1:
            print(
                f"WARNING: Expected 1 object, "
                f"found {len(lines)}: {image_path.name}"
            )
            continue

        parts = lines[0].split()

        if len(parts) < 5:
            print(f"WARNING: Invalid label: {label_path}")
            continue

        class_id = int(parts[0])

        if class_id not in CLASS_NAMES:
            print(
                f"WARNING: Unknown class {class_id}: "
                f"{image_path.name}"
            )
            continue

        xc, yc, bw, bh = map(float, parts[1:5])

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(
                f"WARNING: Could not read "
                f"{image_path.name}: {e}"
            )
            continue

        width, height = image.size

        # ----------------------------------------------------
        # YOLO normalized coordinates -> pixels
        # ----------------------------------------------------

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        # ----------------------------------------------------
        # Clamp coordinates to image boundaries
        # ----------------------------------------------------

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        # ----------------------------------------------------
        # Crop currency note
        # ----------------------------------------------------

        crop = image.crop((x1, y1, x2, y2))

        # ----------------------------------------------------
        # Save into MobileNet class directory
        # ----------------------------------------------------

        class_name = CLASS_NAMES[class_id]

        output_path = (
            MOBILE_DIR
            / split
            / class_name
            / image_path.name
        )

        crop.save(
            output_path,
            quality=95
        )

        counts[split][class_name] += 1
        total += 1

        if index % 500 == 0:
            print(
                f"  Processed {index}/{len(images)} images..."
            )

    print()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("=" * 70)
print("MOBILENETV2 DATASET CREATED")
print("=" * 70)
print()

for split in SPLITS:

    print(f"{split.upper()}:")

    split_total = 0

    for class_id, class_name in CLASS_NAMES.items():

        count = counts[split][class_name]
        split_total += count

        print(
            f"  ₹{class_name:>4}: {count}"
        )

    print(f"  TOTAL: {split_total}")
    print()

print(f"TOTAL CROPPED IMAGES: {total}")
print()
print(f"Dataset location:")
print(MOBILE_DIR.resolve())
print()
print("Original YOLO dataset was NOT modified.")
print("=" * 70)