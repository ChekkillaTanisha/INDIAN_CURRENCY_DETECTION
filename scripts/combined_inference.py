from pathlib import Path
import csv
import shutil

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# Indian Currency YOLO11 + MobileNetV2
# COMBINED INFERENCE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "currency_yolo11n"
    / "weights"
    / "best.pt"
)

MOBILENET_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "mobilenetv2"
    / "best_mobilenetv2.pt"
)

TEST_IMAGES_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "yolo11"
    / "images"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "combined_inference"
)

CROPS_DIR = OUTPUT_DIR / "crops"
CSV_PATH = OUTPUT_DIR / "combined_results.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# IMPORTANT: TWO DIFFERENT CLASS MAPPINGS
# ============================================================

# YOLO11 class order from dataset/yolo11/data.yaml
YOLO_CLASS_NAMES = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
]

# MobileNetV2 was trained using torchvision ImageFolder.
#
# ImageFolder sorts directory names alphabetically/numerically,
# therefore its actual class order is:
#
# 0 = 10
# 1 = 100
# 2 = 20
# 3 = 200
# 4 = 2000
# 5 = 50
# 6 = 500
#
MOBILENET_CLASS_NAMES = [
    "10",
    "100",
    "20",
    "200",
    "2000",
    "50",
    "500",
]


# ============================================================
# MOBILE NET TRANSFORM
# ============================================================

MOBILENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("Indian Currency YOLO11 + MobileNetV2")
print("COMBINED INFERENCE")
print("=" * 70)
print()

print("YOLO11 model:")
print(YOLO_MODEL_PATH)
print()

print("MobileNetV2 model:")
print(MOBILENET_MODEL_PATH)
print()

print("Test images:")
print(TEST_IMAGES_DIR)
print()


# ============================================================
# CHECK FILES
# ============================================================

if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO11 model not found:\n{YOLO_MODEL_PATH}"
    )

if not MOBILENET_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MobileNetV2 model not found:\n{MOBILENET_MODEL_PATH}"
    )

if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_IMAGES_DIR}"
    )


# ============================================================
# LOAD YOLO11
# ============================================================

print("Loading YOLO11...")

yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)

print("YOLO11 loaded.")
print()


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("Loading MobileNetV2...")

checkpoint = torch.load(
    MOBILENET_MODEL_PATH,
    map_location=DEVICE,
)

mobilenet_model = models.mobilenet_v2(
    weights=None
)

mobilenet_model.classifier[1] = nn.Linear(
    mobilenet_model.classifier[1].in_features,
    7,
)

mobilenet_model.load_state_dict(
    checkpoint["model_state_dict"]
)

mobilenet_model = mobilenet_model.to(DEVICE)
mobilenet_model.eval()

print("MobileNetV2 loaded.")
print()


# ============================================================
# TEST IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

image_paths = sorted(
    [
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file()
        and p.suffix in image_extensions
    ]
)

total_images = len(image_paths)

print("=" * 70)
print("RUNNING COMBINED PIPELINE")
print("=" * 70)
print()

print(f"Test images: {total_images}")
print()


# ============================================================
# RESULT STORAGE
# ============================================================

combined_correct = 0
combined_wrong = 0

yolo_detections = 0
yolo_missed = 0

class_correct = {
    name: 0
    for name in YOLO_CLASS_NAMES
}

class_total = {
    name: 0
    for name in YOLO_CLASS_NAMES
}

class_wrong = {
    name: 0
    for name in YOLO_CLASS_NAMES
}

results_rows = []


# ============================================================
# HELPER: GROUND TRUTH FROM IMAGE FILENAME
# ============================================================

def get_ground_truth_from_filename(image_path):
    """
    Dataset filenames begin with the denomination.

    Examples:
        10_18.jpg
        100_70.jpg
        200_96_315.jpg
        500_48_b.jpg

    Return:
        "10", "20", "50", "100", "200", "500", or "2000"
    """

    stem = image_path.stem

    first_part = stem.split("_")[0]

    if first_part in YOLO_CLASS_NAMES:
        return first_part

    return None


# ============================================================
# MAIN LOOP
# ============================================================

for index, image_path in enumerate(image_paths, start=1):

    gt_name = get_ground_truth_from_filename(
        image_path
    )

    if gt_name is None:
        print(
            f"WARNING: Could not determine ground truth: "
            f"{image_path.name}"
        )
        continue

    class_total[gt_name] += 1

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    yolo_results = yolo_model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = yolo_results[0]

    if result.boxes is None or len(result.boxes) == 0:

        yolo_missed += 1

        results_rows.append({
            "image": image_path.name,
            "ground_truth": gt_name,
            "yolo_detected": False,
            "yolo_class": "",
            "yolo_conf": "",
            "mobilenet_prediction": "",
            "mobilenet_conf": "",
            "final_prediction": "",
            "final_correct": False,
        })

        if index % 100 == 0:
            print(
                f"Processed {index}/{total_images} images..."
            )

        continue

    # --------------------------------------------------------
    # SELECT BEST YOLO DETECTION
    # --------------------------------------------------------

    best_box_index = int(
        result.boxes.conf.argmax().item()
    )

    yolo_class_id = int(
        result.boxes.cls[best_box_index].item()
    )

    yolo_conf = float(
        result.boxes.conf[best_box_index].item()
    )

    yolo_class_name = YOLO_CLASS_NAMES[
        yolo_class_id
    ]

    yolo_detections += 1

    # --------------------------------------------------------
    # GET YOLO BOUNDING BOX
    # --------------------------------------------------------

    xyxy = result.boxes.xyxy[
        best_box_index
    ].cpu().numpy()

    x1, y1, x2, y2 = [
        int(round(v))
        for v in xyxy
    ]

    # --------------------------------------------------------
    # OPEN ORIGINAL IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_width, image_height = image.size

    # Clamp coordinates
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(1, min(x2, image_width))
    y2 = max(1, min(y2, image_height))

    # Prevent invalid crop
    if x2 <= x1 or y2 <= y1:

        yolo_missed += 1

        results_rows.append({
            "image": image_path.name,
            "ground_truth": gt_name,
            "yolo_detected": True,
            "yolo_class": yolo_class_name,
            "yolo_conf": f"{yolo_conf:.4f}",
            "mobilenet_prediction": "",
            "mobilenet_conf": "",
            "final_prediction": "",
            "final_correct": False,
        })

        continue

    # --------------------------------------------------------
    # CROP DETECTED NOTE
    # --------------------------------------------------------

    crop = image.crop(
        (x1, y1, x2, y2)
    )

    crop_name = (
        f"{image_path.stem}_crop"
        f"{image_path.suffix}"
    )

    crop_path = CROPS_DIR / crop_name

    crop.save(
        crop_path
    )

    # --------------------------------------------------------
    # MOBILE NET CLASSIFICATION
    # --------------------------------------------------------

    input_tensor = MOBILENET_TRANSFORM(
        crop
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = mobilenet_model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        mobile_prediction_id = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        mobile_conf = float(
            probabilities[0, mobile_prediction_id]
            .item()
        )

    # IMPORTANT:
    # Convert MobileNet's numeric class ID using
    # MobileNet's own class order.
    mobile_prediction_name = (
        MOBILENET_CLASS_NAMES[
            mobile_prediction_id
        ]
    )

    # --------------------------------------------------------
    # FINAL COMBINED PREDICTION
    # --------------------------------------------------------
    #
    # YOLO finds the note.
    # MobileNet classifies the cropped note.
    #
    # For the combined result, use MobileNet's
    # denomination prediction.
    #
    final_prediction = mobile_prediction_name

    correct = (
        final_prediction == gt_name
    )

    if correct:

        combined_correct += 1
        class_correct[gt_name] += 1

    else:

        combined_wrong += 1
        class_wrong[gt_name] += 1

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results_rows.append({
        "image": image_path.name,
        "ground_truth": gt_name,
        "yolo_detected": True,
        "yolo_class": yolo_class_name,
        "yolo_conf": f"{yolo_conf:.4f}",
        "mobilenet_prediction": mobile_prediction_name,
        "mobilenet_conf": f"{mobile_conf:.4f}",
        "final_prediction": final_prediction,
        "final_correct": correct,
    })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 100 == 0:

        print(
            f"Processed {index}/{total_images} images..."
        )


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "image",
    "ground_truth",
    "yolo_detected",
    "yolo_class",
    "yolo_conf",
    "mobilenet_prediction",
    "mobilenet_conf",
    "final_prediction",
    "final_correct",
]

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        results_rows
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("COMBINED PIPELINE RESULTS")
print("=" * 70)
print()

print(
    f"Total test images : {total_images}"
)

print(
    f"YOLO detections   : {yolo_detections}"
)

print(
    f"YOLO missed       : {yolo_missed}"
)

print()

if total_images > 0:

    final_accuracy = (
        combined_correct
        / total_images
        * 100
    )

else:

    final_accuracy = 0.0


print(
    f"FINAL COMBINED ACCURACY: "
    f"{final_accuracy:.2f}%"
)

print()
print("-" * 70)
print("PER-DENOMINATION RESULTS")
print("-" * 70)


for denomination in YOLO_CLASS_NAMES:

    total = class_total[denomination]
    correct = class_correct[denomination]

    if total > 0:
        accuracy = (
            correct
            / total
            * 100
        )
    else:
        accuracy = 0.0

    print(
        f"₹{denomination:>4} : "
        f"{correct:>3}/{total:<3} = "
        f"{accuracy:6.2f}%"
    )


print()
print("-" * 70)
print("ERRORS")
print("-" * 70)

print(
    f"Correct: {combined_correct}"
)

print(
    f"Wrong  : {combined_wrong}"
)

print()
print("Results CSV:")
print(CSV_PATH)

print()
print("Saved crops:")
print(CROPS_DIR)

print()
print("=" * 70)
print("COMBINED INFERENCE COMPLETE")
print("=" * 70)