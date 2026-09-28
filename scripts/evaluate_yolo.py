from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
from ultralytics import YOLO


# ============================================================
# YOLO11 DENOMINATION EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "currency_yolo11n-3"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "yolo11"
    / "images"
    / "test"
)

LABEL_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "yolo11"
    / "labels"
    / "test"
)


CLASS_NAMES = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
]


print()
print("=" * 70)
print("YOLO11 DENOMINATION TEST EVALUATION")
print("=" * 70)
print()

print("Model:")
print(MODEL_PATH)
print()

print("Images:")
print(IMAGE_DIR)
print()

print("Labels:")
print(LABEL_DIR)
print()


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{IMAGE_DIR}"
    )

if not LABEL_DIR.exists():
    raise FileNotFoundError(
        f"Test label directory not found:\n{LABEL_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO11...")

model = YOLO(
    str(MODEL_PATH)
)

print("YOLO11 loaded.")
print()


# ============================================================
# IMAGE LIST
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


image_paths = sorted(
    [
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    ]
)


print(
    f"Test images found: {len(image_paths)}"
)

print()


# ============================================================
# STATISTICS
# ============================================================

total = 0
correct = 0
wrong = 0
missed = 0

confusion = defaultdict(Counter)

per_class_total = Counter()
per_class_correct = Counter()
per_class_wrong = Counter()

wrong_images = []


# ============================================================
# READ GROUND TRUTH
# ============================================================

def get_ground_truth(label_path):

    if not label_path.exists():
        return None

    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    # Take first object.
    parts = lines[0].split()

    if len(parts) < 5:
        return None

    class_id = int(parts[0])

    if (
        class_id < 0
        or class_id >= len(CLASS_NAMES)
    ):
        return None

    return CLASS_NAMES[class_id]


# ============================================================
# EVALUATE
# ============================================================

print("=" * 70)
print("RUNNING TEST")
print("=" * 70)
print()


for index, image_path in enumerate(
    image_paths,
    start=1,
):

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    ground_truth = get_ground_truth(
        label_path
    )

    if ground_truth is None:

        print(
            f"WARNING: No valid label for "
            f"{image_path.name}"
        )

        continue


    total += 1

    per_class_total[
        ground_truth
    ] += 1


    # --------------------------------------------------------
    # YOLO PREDICTION
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),

        conf=0.25,

        iou=0.45,

        imgsz=640,

        device="cpu",

        verbose=False,
    )


    result = results[0]


    # --------------------------------------------------------
    # NO DETECTION
    # --------------------------------------------------------

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        missed += 1
        wrong += 1

        confusion[
            ground_truth
        ]["MISSED"] += 1

        per_class_wrong[
            ground_truth
        ] += 1

        wrong_images.append({
            "image": image_path.name,
            "actual": ground_truth,
            "predicted": "MISSED",
        })

    else:

        # ----------------------------------------------------
        # BEST DETECTION
        # ----------------------------------------------------

        best_index = int(
            result.boxes.conf.argmax().item()
        )

        predicted_id = int(
            result.boxes.cls[
                best_index
            ].item()
        )

        predicted = CLASS_NAMES[
            predicted_id
        ]


        # ----------------------------------------------------
        # CORRECT / WRONG
        # ----------------------------------------------------

        confusion[
            ground_truth
        ][predicted] += 1


        if predicted == ground_truth:

            correct += 1

            per_class_correct[
                ground_truth
            ] += 1

        else:

            wrong += 1

            per_class_wrong[
                ground_truth
            ] += 1

            wrong_images.append({
                "image": image_path.name,
                "actual": ground_truth,
                "predicted": predicted,
            })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 50 == 0:

        print(
            f"Processed "
            f"{index}/{len(image_paths)}..."
        )


# ============================================================
# ACCURACY
# ============================================================

if total > 0:

    accuracy = (
        correct
        / total
        * 100
    )

else:

    accuracy = 0.0


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("YOLO11 TEST RESULTS")
print("=" * 70)
print()

print(
    f"Total images : {total}"
)

print(
    f"Correct      : {correct}"
)

print(
    f"Wrong        : {wrong}"
)

print(
    f"Missed       : {missed}"
)

print()

print(
    f"DENOMINATION ACCURACY: "
    f"{accuracy:.2f}%"
)

print()


# ============================================================
# PER DENOMINATION
# ============================================================

print("=" * 70)
print("PER-DENOMINATION ACCURACY")
print("=" * 70)
print()


for denomination in CLASS_NAMES:

    class_total = per_class_total[
        denomination
    ]

    class_correct = per_class_correct[
        denomination
    ]

    class_wrong = per_class_wrong[
        denomination
    ]


    if class_total > 0:

        class_accuracy = (
            class_correct
            / class_total
            * 100
        )

    else:

        class_accuracy = 0.0


    print(
        f"₹{denomination:>4} : "
        f"{class_correct:>3}/"
        f"{class_total:<3} = "
        f"{class_accuracy:6.2f}% "
        f"(wrong={class_wrong})"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION / ERROR BREAKDOWN")
print("=" * 70)
print()


for actual in CLASS_NAMES:

    row = confusion[
        actual
    ]

    errors = []

    for predicted in CLASS_NAMES:

        count = row[
            predicted
        ]

        if (
            predicted != actual
            and count > 0
        ):

            errors.append(
                f"₹{predicted}={count}"
            )


    missed_count = row[
        "MISSED"
    ]

    if missed_count > 0:

        errors.append(
            f"MISSED={missed_count}"
        )


    if errors:

        print(
            f"Actual ₹{actual}: "
            + ", ".join(errors)
        )

    else:

        print(
            f"Actual ₹{actual}: "
            "NO ERRORS"
        )


# ============================================================
# WRONG IMAGE LIST
# ============================================================

print()
print("=" * 70)
print("INCORRECT IMAGES")
print("=" * 70)
print()


if wrong_images:

    for item in wrong_images:

        print(
            f"{item['image']} | "
            f"Actual=₹{item['actual']} | "
            f"Predicted={item['predicted']}"
        )

else:

    print(
        "No incorrect images."
    )


# ============================================================
# SAVE WRONG IMAGE REPORT
# ============================================================

report_path = (
    PROJECT_ROOT
    / "runs"
    / "currency_yolo11n-3"
    / "wrong_test_predictions.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "YOLO11 WRONG TEST PREDICTIONS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )


    for item in wrong_images:

        file.write(
            f"{item['image']} | "
            f"Actual=₹{item['actual']} | "
            f"Predicted={item['predicted']}\n"
        )


print()
print("Wrong-image report saved to:")
print(report_path)

print()
print("=" * 70)
print("YOLO11 EVALUATION COMPLETE")
print("=" * 70)
print()