from pathlib import Path
import random

import cv2


# ============================================================
# Indian Currency Dataset Visual Verification
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "yolo11"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dataset_preview"

CLASS_NAMES = {
    0: "₹10",
    1: "₹20",
    2: "₹50",
    3: "₹100",
    4: "₹200",
    5: "₹500",
    6: "₹2000",
}

SPLITS = ["train", "val", "test"]

# Number of examples to generate per split.
SAMPLES_PER_SPLIT = 8

RANDOM_SEED = 42


def load_yolo_labels(label_path):
    """
    Read YOLO labels:

        class_id x_center y_center width height

    All coordinates are normalized to [0, 1].
    """

    annotations = []

    if not label_path.exists():
        return annotations

    for line_number, line in enumerate(
        label_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            print(
                f"WARNING: malformed label "
                f"{label_path} line {line_number}"
            )
            continue

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            print(
                f"WARNING: invalid numeric label "
                f"{label_path} line {line_number}"
            )

            continue

        annotations.append(
            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            )
        )

    return annotations


def draw_annotations(image, annotations):
    """
    Convert normalized YOLO boxes to pixel coordinates
    and draw them on the image.
    """

    image_height, image_width = image.shape[:2]

    for (
        class_id,
        x_center,
        y_center,
        width,
        height,
    ) in annotations:

        # Convert normalized coordinates to pixels.

        box_width = width * image_width
        box_height = height * image_height

        center_x = x_center * image_width
        center_y = y_center * image_height

        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)

        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)

        # Keep coordinates inside image.

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))

        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        # Draw bounding box.

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        # Class name.

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}",
        )

        label = f"{class_name} ({class_id})"

        # Determine text size.

        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2,
        )

        # Draw label background.

        label_y1 = max(
            0,
            y1 - text_height - baseline - 5,
        )

        label_y2 = y1

        cv2.rectangle(
            image,
            (x1, label_y1),
            (
                x1 + text_width + 8,
                label_y2,
            ),
            (0, 255, 0),
            -1,
        )

        # Draw text.

        cv2.putText(
            image,
            label,
            (
                x1 + 4,
                y1 - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return image


def create_preview(split):
    """
    Create annotated preview images for one split.
    """

    image_dir = (
        DATASET_ROOT
        / "images"
        / split
    )

    label_dir = (
        DATASET_ROOT
        / "labels"
        / split
    )

    output_dir = (
        OUTPUT_ROOT
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not images:

        print(
            f"WARNING: no images found for {split}"
        )

        return

    random.seed(
        RANDOM_SEED + hash(split)
    )

    sample_count = min(
        SAMPLES_PER_SPLIT,
        len(images),
    )

    selected_images = random.sample(
        images,
        sample_count,
    )

    print()
    print(
        f"{split}: creating "
        f"{sample_count} preview images"
    )

    for index, image_path in enumerate(
        selected_images,
        start=1,
    ):

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: unable to read "
                f"{image_path}"
            )

            continue

        annotations = load_yolo_labels(
            label_path
        )

        annotated_image = draw_annotations(
            image,
            annotations,
        )

        # Add split name to the top-left.

        title = (
            f"{split.upper()} | "
            f"{image_path.name}"
        )

        cv2.putText(
            annotated_image,
            title,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_image,
            title,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        output_path = (
            output_dir
            / f"{index:02d}_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            annotated_image,
        )

        print(
            f"  {index}/{sample_count}: "
            f"{output_path.name}"
        )


def main():

    print("=" * 60)
    print("Indian Currency Dataset Visual Verification")
    print("=" * 60)

    print()
    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_ROOT}"
        )

    for split in SPLITS:
        create_preview(split)

    print()
    print("=" * 60)
    print("VISUAL PREVIEW COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Preview location:\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()