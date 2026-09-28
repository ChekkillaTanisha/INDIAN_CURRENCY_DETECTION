from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v3"
    / "best_authenticity_mobilenetv2_v3.pt"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "authenticity_v3"
    / "validation"
)

DEVICE = torch.device("cpu")


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


DENOMINATIONS = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("V3 DENOMINATION-SPECIFIC THRESHOLD OPTIMIZATION")
print("=" * 70)
print()

print("Model:")
print(MODEL_PATH)
print()

print("Validation:")
print(VALIDATION_DIR)
print()


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model = models.mobilenet_v2(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded.")
print()


# ============================================================
# COLLECT VALIDATION DATA
# ============================================================

samples = []

for class_name in ["fake", "real"]:

    class_dir = VALIDATION_DIR / class_name

    if not class_dir.exists():
        print(
            f"WARNING: missing {class_dir}"
        )
        continue

    for image_path in class_dir.rglob("*"):

        if not (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            continue

        denomination = "unknown"

        for d in DENOMINATIONS:

            if d in [
                part.lower()
                for part in image_path.parts
            ]:
                denomination = d
                break

        samples.append({
            "path": image_path,
            "true_label":
                0
                if class_name == "fake"
                else 1,
            "denomination":
                denomination,
        })


print(
    f"Validation images: {len(samples)}"
)

print()


# ============================================================
# RUN MODEL
# ============================================================

results = []

print("=" * 70)
print("RUNNING MODEL")
print("=" * 70)
print()


with torch.no_grad():

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        try:

            image = Image.open(
                sample["path"]
            ).convert("RGB")

            tensor = (
                transform(image)
                .unsqueeze(0)
                .to(DEVICE)
            )

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1,
            )[0]

            fake_probability = (
                probabilities[0].item()
            )

            results.append({
                "true":
                    sample["true_label"],

                "fake_probability":
                    fake_probability,

                "denomination":
                    sample["denomination"],
            })

        except Exception as e:

            print(
                f"Could not process:"
            )

            print(
                sample["path"]
            )

            print(e)


print(
    f"Successfully processed: "
    f"{len(results)}"
)

print()


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    data,
    threshold,
):

    correct = 0

    fake_correct = 0
    fake_total = 0

    real_correct = 0
    real_total = 0

    for item in data:

        true_label = item["true"]

        fake_probability = (
            item["fake_probability"]
        )

        predicted = (
            0
            if fake_probability >= threshold
            else 1
        )

        if true_label == 0:
            fake_total += 1

            if predicted == 0:
                fake_correct += 1

        else:
            real_total += 1

            if predicted == 1:
                real_correct += 1

        if predicted == true_label:
            correct += 1

    total = len(data)

    accuracy = (
        correct / total
        if total
        else 0
    )

    fake_accuracy = (
        fake_correct / fake_total
        if fake_total
        else 0
    )

    real_accuracy = (
        real_correct / real_total
        if real_total
        else 0
    )

    return (
        accuracy,
        fake_accuracy,
        real_accuracy,
    )


# ============================================================
# GLOBAL THRESHOLD
# ============================================================

print("=" * 70)
print("GLOBAL THRESHOLD")
print("=" * 70)
print()

best_global = None

for i in range(30, 96):

    threshold = i / 100

    accuracy, fake_acc, real_acc = (
        evaluate_threshold(
            results,
            threshold,
        )
    )

    # IMPORTANT:
    # Never sacrifice fake accuracy if avoidable.
    if fake_acc < 0.99:
        continue

    candidate = (
        accuracy,
        real_acc,
        threshold,
    )

    if (
        best_global is None
        or candidate > best_global
    ):
        best_global = candidate


if best_global:

    accuracy, real_acc, threshold = (
        best_global
    )

    _, fake_acc, _ = (
        evaluate_threshold(
            results,
            threshold,
        )
    )

    print(
        f"Threshold      : {threshold:.2f}"
    )

    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"FAKE accuracy  : "
        f"{fake_acc * 100:.2f}%"
    )

    print(
        f"REAL accuracy  : "
        f"{real_acc * 100:.2f}%"
    )

else:

    print(
        "No threshold found that "
        "maintains ≥99% FAKE accuracy."
    )

print()


# ============================================================
# PER-DENOMINATION THRESHOLDS
# ============================================================

print("=" * 70)
print("DENOMINATION-SPECIFIC THRESHOLDS")
print("=" * 70)
print()

threshold_table = {}

for denomination in DENOMINATIONS:

    denomination_data = [
        item
        for item in results
        if item["denomination"]
        == denomination
    ]

    if not denomination_data:

        continue

    best = None

    for i in range(30, 96):

        threshold = i / 100

        accuracy, fake_acc, real_acc = (
            evaluate_threshold(
                denomination_data,
                threshold,
            )
        )

        # Preserve fake detection.
        if fake_acc < 0.99:
            continue

        candidate = (
            accuracy,
            real_acc,
            threshold,
        )

        if (
            best is None
            or candidate > best
        ):
            best = candidate

    if best:

        accuracy, real_acc, threshold = best

        _, fake_acc, _ = (
            evaluate_threshold(
                denomination_data,
                threshold,
            )
        )

        threshold_table[
            denomination
        ] = threshold

        print(
            f"₹{denomination:<5} "
            f"Threshold={threshold:.2f} "
            f"| Accuracy={accuracy * 100:.2f}% "
            f"| FAKE={fake_acc * 100:.2f}% "
            f"| REAL={real_acc * 100:.2f}% "
            f"| N={len(denomination_data)}"
        )

    else:

        threshold_table[
            denomination
        ] = 0.50

        print(
            f"₹{denomination:<5} "
            f"No safe threshold found; "
            f"using 0.50"
        )


print()


# ============================================================
# SIMULATE DENOMINATION-SPECIFIC RESULT
# ============================================================

correct = 0
fake_correct = 0
fake_total = 0
real_correct = 0
real_total = 0

for item in results:

    denomination = item[
        "denomination"
    ]

    threshold = threshold_table.get(
        denomination,
        0.50,
    )

    predicted = (
        0
        if item["fake_probability"]
        >= threshold
        else 1
    )

    if item["true"] == 0:

        fake_total += 1

        if predicted == 0:
            fake_correct += 1

    else:

        real_total += 1

        if predicted == 1:
            real_correct += 1

    if predicted == item["true"]:
        correct += 1


total = len(results)

accuracy = (
    correct / total
    if total
    else 0
)

fake_accuracy = (
    fake_correct / fake_total
    if fake_total
    else 0
)

real_accuracy = (
    real_correct / real_total
    if real_total
    else 0
)


print("=" * 70)
print("VALIDATION SIMULATION")
print("=" * 70)
print()

print(
    f"Accuracy      : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"FAKE accuracy : "
    f"{fake_accuracy * 100:.2f}%"
)

print(
    f"REAL accuracy : "
    f"{real_accuracy * 100:.2f}%"
)

print()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("RECOMMENDED THRESHOLDS")
print("=" * 70)
print()

for denomination in DENOMINATIONS:

    if denomination in threshold_table:

        print(
            f"₹{denomination}: "
            f"{threshold_table[denomination]:.2f}"
        )

print()

print(
    "IMPORTANT:"
)

print(
    "These thresholds were optimized "
    "using VALIDATION data only."
)

print(
    "The TEST set was not used."
)

print()

print(
    "Do NOT change the model yet."
)

print(
    "Use these thresholds for a "
    "fresh test evaluation."
)

print()

print("=" * 70)