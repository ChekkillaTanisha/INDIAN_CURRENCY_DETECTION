from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# FIND AUTHENTICITY MODEL ERRORS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v2"
    / "best_authenticity_mobilenetv2_v2.pt"
)

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "authenticity"
)

DENOMINATIONS = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
]

DEVICE = torch.device("cpu")


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


print()
print("=" * 70)
print("AUTHENTICITY ERROR ANALYSIS")
print("=" * 70)
print()

print("Model:")
print(MODEL_PATH)
print()

print("Dataset:")
print(DATA_DIR)
print()


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MobileNetV2...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
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
# STORAGE
# ============================================================

results = []

confusion = {
    "fake_fake": 0,
    "fake_real": 0,
    "real_fake": 0,
    "real_real": 0,
}


# ============================================================
# FIND ALL IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


for authenticity in ["fake", "real"]:

    root = (
        DATA_DIR
        / authenticity
    )

    if not root.exists():
        continue

    for image_path in root.rglob("*"):

        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in extensions
        ):
            continue

        actual_label = (
            0
            if authenticity == "fake"
            else 1
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = TRANSFORM(
                image
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                output = model(
                    tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                predicted_label = int(
                    probabilities.argmax(
                        dim=1
                    ).item()
                )

                confidence = float(
                    probabilities[
                        0,
                        predicted_label
                    ].item()
                )

            if (
                actual_label == 0
                and predicted_label == 0
            ):

                confusion["fake_fake"] += 1

            elif (
                actual_label == 0
                and predicted_label == 1
            ):

                confusion["fake_real"] += 1

                results.append({
                    "path": str(image_path),
                    "actual": "FAKE",
                    "predicted": "REAL",
                    "confidence": confidence,
                })

            elif (
                actual_label == 1
                and predicted_label == 0
            ):

                confusion["real_fake"] += 1

                results.append({
                    "path": str(image_path),
                    "actual": "REAL",
                    "predicted": "FAKE",
                    "confidence": confidence,
                })

            else:

                confusion["real_real"] += 1

        except Exception as e:

            print(
                f"Could not process: "
                f"{image_path}"
            )

            print(e)


# ============================================================
# RESULTS
# ============================================================

total = sum(
    confusion.values()
)

correct = (
    confusion["fake_fake"]
    + confusion["real_real"]
)

wrong = (
    confusion["fake_real"]
    + confusion["real_fake"]
)


accuracy = (
    correct / total * 100
    if total > 0
    else 0
)


print("=" * 70)
print("AUTHENTICITY RESULTS")
print("=" * 70)
print()

print(
    f"Total images evaluated : {total}"
)

print(
    f"Correct                : {correct}"
)

print(
    f"Wrong                  : {wrong}"
)

print(
    f"Accuracy               : {accuracy:.2f}%"
)

print()

print(
    f"FAKE → FAKE : "
    f"{confusion['fake_fake']}"
)

print(
    f"FAKE → REAL : "
    f"{confusion['fake_real']}"
)

print(
    f"REAL → FAKE : "
    f"{confusion['real_fake']}"
)

print(
    f"REAL → REAL : "
    f"{confusion['real_real']}"
)

print()


# ============================================================
# PRINT ERRORS
# ============================================================

print("=" * 70)
print("MISCLASSIFIED IMAGES")
print("=" * 70)
print()


if not results:

    print("No misclassified images.")

else:

    for index, item in enumerate(
        results,
        start=1,
    ):

        print(
            f"{index}. "
            f"{item['actual']} → "
            f"{item['predicted']} | "
            f"{item['confidence'] * 100:.2f}%"
        )

        print(
            f"   {item['path']}"
        )

        print()


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v2"
    / "misclassified_images.txt"
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "AUTHENTICITY MODEL MISCLASSIFIED IMAGES\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Total: {total}\n"
    )

    file.write(
        f"Correct: {correct}\n"
    )

    file.write(
        f"Wrong: {wrong}\n"
    )

    file.write(
        f"Accuracy: {accuracy:.2f}%\n\n"
    )

    file.write(
        f"FAKE -> FAKE: "
        f"{confusion['fake_fake']}\n"
    )

    file.write(
        f"FAKE -> REAL: "
        f"{confusion['fake_real']}\n"
    )

    file.write(
        f"REAL -> FAKE: "
        f"{confusion['real_fake']}\n"
    )

    file.write(
        f"REAL -> REAL: "
        f"{confusion['real_real']}\n\n"
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        file.write(
            f"{index}. "
            f"{item['actual']} -> "
            f"{item['predicted']} | "
            f"{item['confidence'] * 100:.2f}%\n"
        )

        file.write(
            f"   {item['path']}\n\n"
        )


print("=" * 70)
print("REPORT SAVED")
print("=" * 70)
print()

print(REPORT_PATH)

print()