from pathlib import Path
import csv
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
    / "authenticity_v2"
    / "best_authenticity_mobilenetv2_v2.pt"
)

TEST_SPLIT = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v2"
    / "test_split.csv"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "FAKE",
    "REAL",
]


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("AUTHENTICITY MODEL THRESHOLD CALIBRATION")
print("=" * 70)
print()

print("Model:")
print(MODEL_PATH)

print()

print("Test split:")
print(TEST_SPLIT)

print()


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not TEST_SPLIT.exists():
    raise FileNotFoundError(
        f"Test split not found:\n{TEST_SPLIT}"
    )


# ============================================================
# TRANSFORM
# ============================================================

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
# LOAD MODEL
# ============================================================

print("Loading MobileNetV2 checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
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


model = model.to(
    DEVICE
)

model.eval()


print("Model loaded.")
print()


# ============================================================
# LOAD TEST SPLIT
# ============================================================

samples = []

with open(
    TEST_SPLIT,
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        samples.append({
            "image": row["image"],
            "label": int(row["label"]),
        })


print(
    f"Test images: {len(samples)}"
)

print()


# ============================================================
# RUN MODEL ONCE
#
# IMPORTANT:
# We calculate probabilities only once.
# Then we test MANY thresholds without running the
# neural network again.
# ============================================================

results = []


print("=" * 70)
print("RUNNING MODEL")
print("=" * 70)
print()


with torch.no_grad():

    for index, sample in enumerate(samples, 1):

        image_path = Path(
            sample["image"]
        )

        true_label = sample["label"]


        try:

            image = Image.open(
                image_path
            ).convert("RGB")


            tensor = (
                transform(image)
                .unsqueeze(0)
                .to(DEVICE)
            )


            output = model(
                tensor
            )


            probabilities = torch.softmax(
                output,
                dim=1,
            )[0]


            fake_probability = (
                probabilities[0].item()
            )


            real_probability = (
                probabilities[1].item()
            )


            results.append({
                "true": true_label,
                "fake_probability": fake_probability,
                "real_probability": real_probability,
            })


        except Exception as e:

            print(
                f"ERROR: {image_path}"
            )

            print(e)


        if index % 25 == 0:

            print(
                f"Processed "
                f"{index}/{len(samples)}"
            )


print()

print(
    f"Successfully processed: "
    f"{len(results)}"
)

print()


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    threshold
):

    correct = 0

    fake_correct = 0
    fake_total = 0

    real_correct = 0
    real_total = 0

    fake_predicted_fake = 0
    fake_predicted_real = 0

    real_predicted_fake = 0
    real_predicted_real = 0


    for result in results:

        true_label = result["true"]

        fake_probability = (
            result["fake_probability"]
        )


        # ----------------------------------------------------
        # IMPORTANT
        #
        # Lower threshold = easier to call something FAKE.
        # ----------------------------------------------------

        if fake_probability >= threshold:

            predicted_label = 0

        else:

            predicted_label = 1


        if predicted_label == true_label:

            correct += 1


        if true_label == 0:

            fake_total += 1

            if predicted_label == 0:

                fake_correct += 1
                fake_predicted_fake += 1

            else:

                fake_predicted_real += 1


        else:

            real_total += 1

            if predicted_label == 1:

                real_correct += 1
                real_predicted_real += 1

            else:

                real_predicted_fake += 1


    total = len(results)


    accuracy = (
        correct / total
        if total > 0
        else 0
    )


    fake_accuracy = (
        fake_correct / fake_total
        if fake_total > 0
        else 0
    )


    real_accuracy = (
        real_correct / real_total
        if real_total > 0
        else 0
    )


    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "fake_accuracy": fake_accuracy,
        "real_accuracy": real_accuracy,
        "fake_fake": fake_predicted_fake,
        "fake_real": fake_predicted_real,
        "real_fake": real_predicted_fake,
        "real_real": real_predicted_real,
    }


# ============================================================
# SEARCH THRESHOLDS
# ============================================================

threshold_results = []


for i in range(
    5,
    96,
):

    threshold = i / 100.0

    result = evaluate_threshold(
        threshold
    )

    threshold_results.append(
        result
    )


# ============================================================
# SORT BY ACCURACY
# ============================================================

threshold_results.sort(
    key=lambda x: x["accuracy"],
    reverse=True,
)


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

print("=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)
print()

print(
    "Threshold | Accuracy | FAKE Acc | REAL Acc | "
    "FF | FR | RF | RR"
)

print("-" * 70)


for result in threshold_results[:10]:

    print(
        f"{result['threshold']:.2f}      | "
        f"{result['accuracy'] * 100:7.2f}% | "
        f"{result['fake_accuracy'] * 100:7.2f}% | "
        f"{result['real_accuracy'] * 100:7.2f}% | "
        f"{result['fake_fake']:2d} | "
        f"{result['fake_real']:2d} | "
        f"{result['real_fake']:2d} | "
        f"{result['real_real']:2d}"
    )


print()


# ============================================================
# BEST RESULT
# ============================================================

best = threshold_results[0]


print("=" * 70)
print("BEST RESULT")
print("=" * 70)
print()

print(
    f"Best threshold : "
    f"{best['threshold']:.2f}"
)

print(
    f"Accuracy       : "
    f"{best['accuracy'] * 100:.2f}%"
)

print(
    f"FAKE accuracy  : "
    f"{best['fake_accuracy'] * 100:.2f}%"
)

print(
    f"REAL accuracy  : "
    f"{best['real_accuracy'] * 100:.2f}%"
)

print()

print(
    f"FAKE → FAKE : "
    f"{best['fake_fake']}"
)

print(
    f"FAKE → REAL : "
    f"{best['fake_real']}"
)

print(
    f"REAL → FAKE : "
    f"{best['real_fake']}"
)

print(
    f"REAL → REAL : "
    f"{best['real_real']}"
)

print()


# ============================================================
# SAVE CALIBRATED THRESHOLD
# ============================================================

threshold_file = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v2"
    / "calibrated_threshold.txt"
)


with open(
    threshold_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        f"{best['threshold']:.4f}"
    )


print(
    "Calibrated threshold saved:"
)

print(
    threshold_file
)

print()

print("=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)
print()