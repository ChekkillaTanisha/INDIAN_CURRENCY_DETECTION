from pathlib import Path
import random
import shutil
from collections import Counter, defaultdict

# ============================================================
# Indian Currency YOLO11 Dataset Preparation
# ============================================================
# SOURCE DATASET:
# C:\Users\Tanisha\Downloads\IndianCurrency_Kaggle
#
# OUTPUT:
# <project>\dataset\yolo11
#
# IMPORTANT:
# - Original Kaggle dataset is NEVER modified.
# - Augmented variants are kept in the same split.
# - Invalid YOLO boxes are removed.
# - A report is generated so every decision is traceable.
# ============================================================


# -----------------------------
# Configuration
# -----------------------------

SOURCE_ROOT = Path(
    r"C:\Users\Tanisha\Downloads\IndianCurrency_Kaggle"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "yolo11"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Reproducible split
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# YOLO classes
CLASS_NAMES = {
    0: "10",
    1: "20",
    2: "50",
    3: "100",
    4: "200",
    5: "500",
    6: "2000",
}

DENOMINATION_TO_CLASS = {
    "10": 0,
    "20": 1,
    "50": 2,
    "100": 3,
    "200": 4,
    "500": 5,
    "2000": 6,
}

# Dataset augmentation suffixes found during inspection.
# These variants must stay together with their base image.
AUGMENTATION_SUFFIXES = (
    "_45",
    "_315",
    "_b",
)


# -----------------------------
# Utility functions
# -----------------------------

def get_base_name(stem: str) -> str:
    """
    Convert:
        10_100
        10_100_45
        10_100_315
        10_100_b

    into the same base group:
        10_100
    """

    for suffix in AUGMENTATION_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    return stem


def find_label_for_image(image_path: Path) -> Path:
    """
    Labels are stored in a separate *Annotations folder.

    Example:
        10/10Rupees/10_1.jpg
        10/10RupeesAnnotations/10_1.txt
    """

    denomination = image_path.parent.parent.name

    annotation_dir = (
        image_path.parent.parent
        / f"{denomination}RupeesAnnotations"
    )

    label_path = annotation_dir / f"{image_path.stem}.txt"

    return label_path


def read_and_validate_label(
    label_path: Path,
    expected_class: int,
):
    """
    Read YOLO annotation file.

    Valid YOLO line:
        class x_center y_center width height

    Invalid boxes are removed.

    A box is considered invalid when:
        width <= 0
        height <= 0
        coordinates are outside [0, 1]
        malformed line
        class does not match expected denomination
    """

    valid_lines = []
    invalid_lines = []

    if not label_path.exists():
        return valid_lines, ["MISSING_LABEL"]

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()
    except Exception as exc:
        return valid_lines, [f"READ_ERROR: {exc}"]

    for line_number, raw_line in enumerate(lines, start=1):

        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            invalid_lines.append(
                f"line {line_number}: expected 5 values"
            )
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            invalid_lines.append(
                f"line {line_number}: non-numeric value"
            )
            continue

        # Class must correspond to denomination folder.
        if class_id != expected_class:
            invalid_lines.append(
                f"line {line_number}: "
                f"class {class_id} != expected {expected_class}"
            )
            continue

        # YOLO normalized coordinates must be in [0, 1].
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 <= width <= 1.0
            and 0.0 <= height <= 1.0
        ):
            invalid_lines.append(
                f"line {line_number}: coordinate outside [0,1]"
            )
            continue

        # Width and height must be strictly positive.
        if width <= 0.0 or height <= 0.0:
            invalid_lines.append(
                f"line {line_number}: "
                f"non-positive box size "
                f"(w={width}, h={height})"
            )
            continue

        # Store a normalized representation.
        valid_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    return valid_lines, invalid_lines


def make_output_dirs():
    """Create YOLO train/val/test directories."""

    for split in ("train", "val", "test"):

        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


def collect_images():
    """
    Collect all images from denomination folders.

    Expected structure:

        10/10Rupees/*.jpg
        20/20Rupees/*.jpg
        ...
    """

    records = []

    for denomination in DENOMINATION_TO_CLASS:

        denomination_dir = (
            SOURCE_ROOT / denomination
        )

        image_dir = (
            denomination_dir
            / f"{denomination}Rupees"
        )

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found:\n{image_dir}"
            )

        expected_class = DENOMINATION_TO_CLASS[
            denomination
        ]

        for image_path in sorted(image_dir.iterdir()):

            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            label_path = find_label_for_image(
                image_path
            )

            base_name = get_base_name(
                image_path.stem
            )

            # Group is denomination + base image name.
            group_id = (
                f"{denomination}__{base_name}"
            )

            records.append(
                {
                    "image": image_path,
                    "label": label_path,
                    "denomination": denomination,
                    "class_id": expected_class,
                    "group_id": group_id,
                }
            )

    return records


def split_groups(records):
    """
    Split by image groups, NOT individual images.

    This prevents augmented versions of the same
    source image from leaking across train/val/test.
    """

    groups = defaultdict(list)

    for record in records:
        groups[record["group_id"]].append(record)

    group_ids = list(groups.keys())

    random.seed(RANDOM_SEED)
    random.shuffle(group_ids)

    total_groups = len(group_ids)

    train_count = int(
        total_groups * TRAIN_RATIO
    )

    val_count = int(
        total_groups * VAL_RATIO
    )

    train_groups = set(
        group_ids[:train_count]
    )

    val_groups = set(
        group_ids[
            train_count:
            train_count + val_count
        ]
    )

    test_groups = set(
        group_ids[
            train_count + val_count:
        ]
    )

    split_map = {}

    for group_id in train_groups:
        split_map[group_id] = "train"

    for group_id in val_groups:
        split_map[group_id] = "val"

    for group_id in test_groups:
        split_map[group_id] = "test"

    return groups, split_map


def copy_dataset(records, split_map):
    """
    Validate labels and copy images/labels.

    Invalid annotation rows are removed.

    If an image has no valid boxes after cleaning,
    it is excluded from the detection dataset.
    """

    statistics = {
        "images_seen": 0,
        "images_copied": 0,
        "images_skipped": 0,
        "labels_written": 0,
        "invalid_boxes": 0,
        "missing_labels": 0,
        "class_mismatch": 0,
    }

    class_image_counts = Counter()
    class_box_counts = Counter()

    invalid_report = []

    for index, record in enumerate(records, start=1):

        statistics["images_seen"] += 1

        image_path = record["image"]
        label_path = record["label"]
        expected_class = record["class_id"]

        split = split_map[record["group_id"]]

        valid_lines, invalid_lines = (
            read_and_validate_label(
                label_path,
                expected_class
            )
        )

        if not label_path.exists():
            statistics["missing_labels"] += 1

        statistics["invalid_boxes"] += len(
            invalid_lines
        )

        if invalid_lines:
            for issue in invalid_lines:
                invalid_report.append(
                    f"{image_path}\t{issue}"
                )

        # If no valid boxes remain, don't train on this image.
        if not valid_lines:

            statistics["images_skipped"] += 1

            continue

        output_image = (
            OUTPUT_ROOT
            / "images"
            / split
            / image_path.name
        )

        output_label = (
            OUTPUT_ROOT
            / "labels"
            / split
            / f"{image_path.stem}.txt"
        )

        # Avoid accidental filename collisions.
        if output_image.exists():
            raise RuntimeError(
                "Filename collision detected:\n"
                f"{output_image}"
            )

        if output_label.exists():
            raise RuntimeError(
                "Label filename collision detected:\n"
                f"{output_label}"
            )

        shutil.copy2(
            image_path,
            output_image
        )

        output_label.write_text(
            "\n".join(valid_lines) + "\n",
            encoding="utf-8"
        )

        statistics["images_copied"] += 1
        statistics["labels_written"] += 1

        class_image_counts[
            expected_class
        ] += 1

        class_box_counts[
            expected_class
        ] += len(valid_lines)

        if index % 250 == 0:
            print(
                f"Processed {index}/{len(records)} images..."
            )

    return (
        statistics,
        class_image_counts,
        class_box_counts,
        invalid_report,
    )


def write_data_yaml():
    """Create YOLO dataset configuration."""

    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}

names:
"""

    for class_id in sorted(CLASS_NAMES):
        yaml_text += (
            f"  {class_id}: '{CLASS_NAMES[class_id]}'\n"
        )

    yaml_path = OUTPUT_ROOT / "data.yaml"

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    return yaml_path


def write_report(
    records,
    split_map,
    statistics,
    class_image_counts,
    class_box_counts,
    invalid_report,
):
    """Write a complete preparation report."""

    report_path = (
        OUTPUT_ROOT / "dataset_report.txt"
    )

    groups = defaultdict(list)

    for record in records:
        groups[record["group_id"]].append(record)

    split_group_counts = Counter(
        split_map.values()
    )

    split_image_counts = Counter()

    for record in records:
        split_image_counts[
            split_map[record["group_id"]]
        ] += 1

    lines = []

    lines.append(
        "Indian Currency YOLO11 Dataset Report"
    )
    lines.append("=" * 55)
    lines.append("")

    lines.append(
        f"Source: {SOURCE_ROOT}"
    )
    lines.append(
        f"Output: {OUTPUT_ROOT}"
    )
    lines.append(
        f"Random seed: {RANDOM_SEED}"
    )
    lines.append("")

    lines.append(
        "SOURCE DATASET"
    )
    lines.append("-" * 55)

    lines.append(
        f"Images discovered: {len(records)}"
    )

    lines.append(
        f"Unique image groups: {len(groups)}"
    )

    lines.append("")

    lines.append(
        "GROUP SPLIT"
    )
    lines.append("-" * 55)

    for split in ("train", "val", "test"):
        lines.append(
            f"{split}: "
            f"{split_group_counts[split]} groups, "
            f"{split_image_counts[split]} images"
        )

    lines.append("")

    lines.append(
        "FINAL DATASET"
    )
    lines.append("-" * 55)

    for key, value in statistics.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append(
        "CLASS DISTRIBUTION"
    )
    lines.append("-" * 55)

    for class_id in sorted(CLASS_NAMES):

        lines.append(
            f"class {class_id} "
            f"(₹{CLASS_NAMES[class_id]}): "
            f"{class_image_counts[class_id]} images, "
            f"{class_box_counts[class_id]} boxes"
        )

    lines.append("")

    lines.append(
        "INVALID ANNOTATION DETAILS"
    )
    lines.append("-" * 55)

    if invalid_report:

        for item in invalid_report:
            lines.append(item)

    else:

        lines.append(
            "No invalid annotations found."
        )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return report_path


# -----------------------------
# Main
# -----------------------------

def main():

    print("=" * 60)
    print("Indian Currency YOLO11 Dataset Preparation")
    print("=" * 60)

    print()
    print(f"Source: {SOURCE_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print()

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source dataset does not exist:\n"
            f"{SOURCE_ROOT}"
        )

    if OUTPUT_ROOT.exists():

        existing_files = list(
            OUTPUT_ROOT.rglob("*")
        )

        if existing_files:

            raise RuntimeError(
                "Output dataset directory is not empty:\n"
                f"{OUTPUT_ROOT}\n\n"
                "This script refuses to overwrite an "
                "existing dataset."
            )

    make_output_dirs()

    print("Collecting images...")

    records = collect_images()

    print(
        f"Images discovered: {len(records)}"
    )

    if len(records) == 0:
        raise RuntimeError(
            "No images were found."
        )

    print()
    print("Creating group-aware split...")

    groups, split_map = split_groups(records)

    print(
        f"Unique groups: {len(groups)}"
    )

    for split in ("train", "val", "test"):

        group_count = sum(
            1
            for group_id in groups
            if split_map[group_id] == split
        )

        print(
            f"{split}: {group_count} groups"
        )

    print()
    print("Validating labels and copying dataset...")
    print()

    (
        statistics,
        class_image_counts,
        class_box_counts,
        invalid_report,
    ) = copy_dataset(
        records,
        split_map
    )

    print()
    print("Creating data.yaml...")

    yaml_path = write_data_yaml()

    print(
        f"Created: {yaml_path}"
    )

    print()
    print("Writing dataset report...")

    report_path = write_report(
        records,
        split_map,
        statistics,
        class_image_counts,
        class_box_counts,
        invalid_report,
    )

    print(
        f"Created: {report_path}"
    )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Images copied: {statistics['images_copied']}"
    )

    print(
        f"Images skipped: {statistics['images_skipped']}"
    )

    print(
        f"Invalid annotation rows removed: "
        f"{statistics['invalid_boxes']}"
    )

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    print()
    print("Next step will be dataset verification.")
    print("DO NOT TRAIN YOLO11 YET.")


if __name__ == "__main__":
    main()