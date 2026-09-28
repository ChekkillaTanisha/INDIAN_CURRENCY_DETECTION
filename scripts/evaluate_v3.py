from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v3"
    / "best_authenticity_mobilenetv2_v3.pt"
)

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "authenticity_v3"
    / "test"
)

DEVICE = torch.device("cpu")

CLASS_NAMES = ["FAKE", "REAL"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

print("=" * 70)
print("V3 AUTHENTICITY MODEL TEST")
print("=" * 70)
print()

print("Model:")
print(MODEL_PATH)
print()

print("Test dataset:")
print(TEST_DIR)
print()

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model = models.mobilenet_v2(weights=None)

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

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

confusion = {
    "fake_fake": 0,
    "fake_real": 0,
    "real_fake": 0,
    "real_real": 0,
}

total = 0
correct = 0

for class_name, true_label in [
    ("fake", 0),
    ("real", 1),
]:

    class_dir = TEST_DIR / class_name

    images = []

    for p in class_dir.rglob("*"):
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }:
            images.append(p)

    print(
        f"{class_name.upper()} images: {len(images)}"
    )

    for image_path in images:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = transform(
                image
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                output = model(image)

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

                predicted = (
                    probabilities
                    .argmax(dim=1)
                    .item()
                )

            total += 1

            if predicted == true_label:
                correct += 1

            if true_label == 0 and predicted == 0:
                confusion["fake_fake"] += 1

            elif true_label == 0 and predicted == 1:
                confusion["fake_real"] += 1

            elif true_label == 1 and predicted == 0:
                confusion["real_fake"] += 1

            elif true_label == 1 and predicted == 1:
                confusion["real_real"] += 1

        except Exception as e:

            print(
                f"ERROR: {image_path}"
            )

            print(e)

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

accuracy = correct / total

fake_total = (
    confusion["fake_fake"]
    + confusion["fake_real"]
)

real_total = (
    confusion["real_fake"]
    + confusion["real_real"]
)

fake_accuracy = (
    confusion["fake_fake"]
    / fake_total
)

real_accuracy = (
    confusion["real_real"]
    / real_total
)

print()
print("=" * 70)
print("V3 TEST RESULT")
print("=" * 70)
print()

print(f"Correct : {correct}/{total}")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()

print(
    f"FAKE accuracy: "
    f"{fake_accuracy * 100:.2f}%"
)

print(
    f"REAL accuracy: "
    f"{real_accuracy * 100:.2f}%"
)

print()

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print()

print(
    f"FAKE -> FAKE : "
    f"{confusion['fake_fake']}"
)

print(
    f"FAKE -> REAL : "
    f"{confusion['fake_real']}"
)

print(
    f"REAL -> FAKE : "
    f"{confusion['real_fake']}"
)

print(
    f"REAL -> REAL : "
    f"{confusion['real_real']}"
)

print()