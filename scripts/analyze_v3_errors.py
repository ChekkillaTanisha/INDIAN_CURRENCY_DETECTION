from pathlib import Path
from collections import Counter

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

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "authenticity_v3"
    / "test"
)

DEVICE = torch.device("cpu")


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = [
    "FAKE",
    "REAL",
]

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
# HEADER
# ============================================================

print()
print("=" * 70)
print("V3 AUTHENTICITY ERROR ANALYSIS")
print("=" * 70)
print()

print("Model:")
print(MODEL_PATH)
print()

print("Test dataset:")
print(TEST_DIR)
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)
print()

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
# COLLECT ERRORS
# ============================================================

errors = []

total = 0
correct = 0

confusion = {
    "fake_fake": 0,
    "fake_real": 0,
    "real_fake": 0,
    "real_real": 0,
}


# ============================================================
# PROCESS DATASET
# ============================================================

for true_class in [
    "fake",
    "real",
]:

    class_dir = TEST_DIR / true_class

    if not class_dir.exists():
        print(
            f"WARNING: missing folder: {class_dir}"
        )
        continue

    image_paths = [
        p
        for p in class_dir.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    true_label = (
        0
        if true_class == "fake"
        else 1
    )

    print(
        f"Processing {true_class.upper()}: "
        f"{len(image_paths)} images"
    )

    for image_path in image_paths:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = (
                transform(image)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():

                output = model(tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )[0]

            predicted_label = (
                probabilities
                .argmax()
                .item()
            )

            confidence = (
                probabilities[
                    predicted_label
                ].item()
            )

            real_probability = (
                probabilities[1].item()
            )

            fake_probability = (
                probabilities[0].item()
            )

            total += 1

            if predicted_label == true_label:

                correct += 1

                if true_label == 0:
                    confusion[
                        "fake_fake"
                    ] += 1

                else:
                    confusion[
                        "real_real"
                    ] += 1

            else:

                if true_label == 0:

                    confusion[
                        "fake_real"
                    ] += 1

                else:

                    confusion[
                        "real_fake"
                    ] += 1

                # ------------------------------------------------
                # TRY TO DETECT DENOMINATION FROM PATH
                # ------------------------------------------------

                denomination = "UNKNOWN"

                parts_lower = [
                    part.lower()
                    for part in image_path.parts
                ]

                for denomination_name in [
                    "10",
                    "20",
                    "50",
                    "100",
                    "200",
                    "500",
                    "2000",
                ]:

                    if (
                        denomination_name
                        in parts_lower
                    ):

                        denomination = (
                            f"₹{denomination_name}"
                        )

                        break

                errors.append(
                    {
                        "path": str(
                            image_path
                        ),

                        "true": CLASS_NAMES[
                            true_label
                        ],

                        "predicted": CLASS_NAMES[
                            predicted_label
                        ],

                        "confidence": confidence,

                        "fake_probability":
                            fake_probability,

                        "real_probability":
                            real_probability,

                        "denomination":
                            denomination,
                    }
                )

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


# ============================================================
# OVERALL RESULT
# ============================================================

accuracy = (
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
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()


# ============================================================
# CONFUSION
# ============================================================

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


# ============================================================
# ERRORS
# ============================================================

print("=" * 70)
print("INCORRECT IMAGES")
print("=" * 70)
print()

if not errors:

    print(
        "🎉 NO ERRORS!"
    )

else:

    # --------------------------------------------------------
    # Sort by confidence
    # Highest-confidence mistakes first
    # --------------------------------------------------------

    errors_sorted = sorted(
        errors,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    for i, error in enumerate(
        errors_sorted,
        start=1,
    ):

        print(
            f"[{i:02d}] "
            f"{error['true']} -> "
            f"{error['predicted']} "
            f"| Confidence: "
            f"{error['confidence'] * 100:.2f}% "
            f"| "
            f"Fake P="
            f"{error['fake_probability'] * 100:.2f}% "
            f"| "
            f"Real P="
            f"{error['real_probability'] * 100:.2f}%"
        )

        print(
            f"     Denomination: "
            f"{error['denomination']}"
        )

        print(
            f"     {error['path']}"
        )

        print()


# ============================================================
# DENOMINATION ANALYSIS
# ============================================================

print("=" * 70)
print("ERRORS BY DENOMINATION")
print("=" * 70)
print()

denomination_counts = Counter(
    error["denomination"]
    for error in errors
)

if denomination_counts:

    for denomination, count in sorted(
        denomination_counts.items()
    ):

        print(
            f"{denomination:<10}: {count}"
        )

else:

    print(
        "No errors."
    )

print()


# ============================================================
# REAL -> FAKE ANALYSIS
# ============================================================

real_fake_errors = [
    error
    for error in errors
    if (
        error["true"] == "REAL"
        and error["predicted"] == "FAKE"
    )
]

print("=" * 70)
print("REAL -> FAKE ERRORS")
print("=" * 70)
print()

print(
    f"Count: "
    f"{len(real_fake_errors)}"
)

print()

for i, error in enumerate(
    sorted(
        real_fake_errors,
        key=lambda x: x["confidence"],
        reverse=True,
    ),
    start=1,
):

    print(
        f"{i:02d}. "
        f"{error['denomination']} "
        f"| Fake confidence: "
        f"{error['fake_probability'] * 100:.2f}%"
    )

    print(
        f"    Real probability: "
        f"{error['real_probability'] * 100:.2f}%"
    )

    print(
        f"    {error['path']}"
    )

    print()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print()

print(
    "No training was performed."
)

print(
    "No model file was modified."
)

print()