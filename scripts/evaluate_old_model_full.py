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

OLD_DATASET = Path(
    r"C:\Users\Tanisha\OneDrive\Desktop\VERINOTE\ai\datasets\raw\Indian_Currency_Real_vs_Fake"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v2"
    / "full_dataset_evaluation.csv"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


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
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("FULL ORIGINAL DATASET EVALUATION")
print("=" * 70)
print()

print("Model:")
print(MODEL_PATH)

print()

print("Original dataset:")
print(OLD_DATASET)

print()


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not OLD_DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{OLD_DATASET}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING OLD MODEL")
print("=" * 70)
print()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)


print(
    "Checkpoint information:"
)

print(
    f"Architecture      : "
    f"{checkpoint.get('architecture')}"
)

print(
    f"Classes            : "
    f"{checkpoint.get('class_names')}"
)

print(
    f"Best val accuracy  : "
    f"{checkpoint.get('best_val_accuracy', 0) * 100:.2f}%"
)

print(
    f"Old test accuracy  : "
    f"{checkpoint.get('test_accuracy', 0) * 100:.2f}%"
)

print()


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


print("Model loaded successfully.")
print()


# ============================================================
# COLLECT ORIGINAL DATASET
# ============================================================

print("=" * 70)
print("COLLECTING ORIGINAL DATASET")
print("=" * 70)
print()


samples = []


for class_name, label in [
    ("fake", 0),
    ("real", 1),
]:

    class_dir = OLD_DATASET / class_name


    if not class_dir.exists():

        raise FileNotFoundError(
            f"Missing folder:\n{class_dir}"
        )


    for denomination_dir in class_dir.iterdir():

        if not denomination_dir.is_dir():
            continue


        denomination = denomination_dir.name


        for image_path in denomination_dir.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                samples.append({
                    "path": image_path,
                    "label": label,
                    "class_name": class_name,
                    "denomination": denomination,
                })


print(
    f"Total images found: {len(samples)}"
)

print()


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

print("=" * 70)
print("DATASET DISTRIBUTION")
print("=" * 70)
print()


denominations = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
]


for denomination in denominations:

    fake = sum(
        1
        for s in samples
        if s["denomination"] == denomination
        and s["label"] == 0
    )

    real = sum(
        1
        for s in samples
        if s["denomination"] == denomination
        and s["label"] == 1
    )

    print(
        f"₹{denomination:<4} "
        f"FAKE={fake:<5} "
        f"REAL={real:<5} "
        f"TOTAL={fake + real}"
    )


print()


# ============================================================
# EVALUATION STORAGE
# ============================================================

results = []


# ============================================================
# EVALUATE
# ============================================================

print("=" * 70)
print("RUNNING MODEL")
print("=" * 70)
print()

correct = 0


with torch.no_grad():

    for index, sample in enumerate(
        samples,
        1,
    ):

        image_path = sample["path"]

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


            predicted_label = (
                0
                if fake_probability >= 0.50
                else 1
            )


            if predicted_label == true_label:

                correct += 1


            results.append({
                "image": str(image_path),
                "denomination":
                    sample["denomination"],
                "true_label":
                    "FAKE"
                    if true_label == 0
                    else "REAL",
                "predicted_label":
                    "FAKE"
                    if predicted_label == 0
                    else "REAL",
                "fake_probability":
                    fake_probability,
                "real_probability":
                    real_probability,
                "correct":
                    predicted_label == true_label,
            })


        except Exception as e:

            print()
            print(
                "ERROR:"
            )

            print(
                image_path
            )

            print(
                e
            )


        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/{len(samples)}"
            )


# ============================================================
# OVERALL ACCURACY
# ============================================================

total = len(results)


overall_accuracy = (
    correct / total
    if total > 0
    else 0
)


print()
print("=" * 70)
print("OVERALL RESULT")
print("=" * 70)
print()

print(
    f"Correct : {correct}/{total}"
)

print(
    f"Accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print()


# ============================================================
# FAKE / REAL PERFORMANCE
# ============================================================

for label_name in [
    "FAKE",
    "REAL",
]:

    class_results = [
        r
        for r in results
        if r["true_label"] == label_name
    ]


    class_correct = sum(
        1
        for r in class_results
        if r["correct"]
    )


    class_accuracy = (
        class_correct / len(class_results)
        if class_results
        else 0
    )


    print(
        f"{label_name} accuracy: "
        f"{class_accuracy * 100:.2f}% "
        f"({class_correct}/{len(class_results)})"
    )


print()


# ============================================================
# DENOMINATION PERFORMANCE
# ============================================================

print("=" * 70)
print("DENOMINATION PERFORMANCE")
print("=" * 70)
print()

print(
    "Denomination | Accuracy | Correct | Total | "
    "FAKE Acc | REAL Acc"
)

print("-" * 70)


for denomination in denominations:

    denomination_results = [
        r
        for r in results
        if r["denomination"] == denomination
    ]


    denomination_correct = sum(
        1
        for r in denomination_results
        if r["correct"]
    )


    denomination_total = len(
        denomination_results
    )


    denomination_accuracy = (
        denomination_correct
        /
        denomination_total
        if denomination_total
        else 0
    )


    fake_results = [
        r
        for r in denomination_results
        if r["true_label"] == "FAKE"
    ]


    real_results = [
        r
        for r in denomination_results
        if r["true_label"] == "REAL"
    ]


    fake_correct = sum(
        1
        for r in fake_results
        if r["correct"]
    )


    real_correct = sum(
        1
        for r in real_results
        if r["correct"]
    )


    fake_accuracy = (
        fake_correct / len(fake_results)
        if fake_results
        else 0
    )


    real_accuracy = (
        real_correct / len(real_results)
        if real_results
        else 0
    )


    print(
        f"₹{denomination:<12} | "
        f"{denomination_accuracy * 100:7.2f}% | "
        f"{denomination_correct:7d} | "
        f"{denomination_total:5d} | "
        f"{fake_accuracy * 100:7.2f}% | "
        f"{real_accuracy * 100:7.2f}%"
    )


print()


# ============================================================
# THRESHOLD SEARCH ON ENTIRE DATASET
# ============================================================

print("=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)
print()


threshold_results = []


for i in range(
    20,
    81,
):

    threshold = i / 100.0


    threshold_correct = 0

    fake_correct = 0
    fake_total = 0

    real_correct = 0
    real_total = 0


    for result in results:

        predicted = (
            0
            if result["fake_probability"] >= threshold
            else 1
        )


        true_label = (
            0
            if result["true_label"] == "FAKE"
            else 1
        )


        if predicted == true_label:

            threshold_correct += 1


        if true_label == 0:

            fake_total += 1

            if predicted == 0:

                fake_correct += 1


        else:

            real_total += 1

            if predicted == 1:

                real_correct += 1


    accuracy = (
        threshold_correct / total
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


    # Balanced accuracy gives equal importance
    # to FAKE and REAL.

    balanced_accuracy = (
        fake_accuracy
        +
        real_accuracy
    ) / 2


    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "fake_accuracy": fake_accuracy,
        "real_accuracy": real_accuracy,
        "balanced_accuracy":
            balanced_accuracy,
    })


# ------------------------------------------------------------
# BEST OVERALL ACCURACY
# ------------------------------------------------------------

best_accuracy = max(
    threshold_results,
    key=lambda x: x["accuracy"]
)


# ------------------------------------------------------------
# BEST BALANCED ACCURACY
# ------------------------------------------------------------

best_balanced = max(
    threshold_results,
    key=lambda x: x["balanced_accuracy"]
)


print(
    "BEST OVERALL ACCURACY"
)

print(
    f"Threshold : "
    f"{best_accuracy['threshold']:.2f}"
)

print(
    f"Accuracy  : "
    f"{best_accuracy['accuracy'] * 100:.2f}%"
)

print(
    f"FAKE      : "
    f"{best_accuracy['fake_accuracy'] * 100:.2f}%"
)

print(
    f"REAL      : "
    f"{best_accuracy['real_accuracy'] * 100:.2f}%"
)

print()


print(
    "BEST BALANCED ACCURACY"
)

print(
    f"Threshold : "
    f"{best_balanced['threshold']:.2f}"
)

print(
    f"Balanced  : "
    f"{best_balanced['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"FAKE      : "
    f"{best_balanced['fake_accuracy'] * 100:.2f}%"
)

print(
    f"REAL      : "
    f"{best_balanced['real_accuracy'] * 100:.2f}%"
)

print()


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "denomination",
        "true_label",
        "predicted_label",
        "fake_probability",
        "real_probability",
        "correct",
    ])


    for result in results:

        writer.writerow([
            result["image"],
            result["denomination"],
            result["true_label"],
            result["predicted_label"],
            f"{result['fake_probability']:.6f}",
            f"{result['real_probability']:.6f}",
            result["correct"],
        ])


print(
    "Detailed results saved:"
)

print(
    OUTPUT_PATH
)

print()

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print()