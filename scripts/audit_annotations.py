from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import math

# ============================================================
# Indian Currency YOLO Annotation Audit
# READ-ONLY — DOES NOT MODIFY DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "dataset" / "yolo11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "annotation_audit"

SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "10",
    1: "20",
    2: "50",
    3: "100",
    4: "200",
    5: "500",
    6: "2000",
}

# Thresholds are for FLAGGING only.
# They do NOT automatically mean an annotation is wrong.
VERY_SMALL_BOX_AREA = 0.01       # < 1% of image
VERY_LARGE_BOX_AREA = 0.95       # > 95% of image
BORDER_EPS = 0.005               # within 0.5% of image boundary
MIN_BOX_SIZE = 0.001             # extremely tiny width/height


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def audit_label(label_path, image_path):
    """
    Returns:
        rows: valid annotation rows
        problems: list of detected problems
    """

    rows = []
    problems = []

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        return rows, [f"IMAGE_READ_ERROR: {e}"]

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return rows, [f"LABEL_READ_ERROR: {e}"]

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            problems.append(
                f"LINE_FORMAT line={line_number} values={len(parts)}"
            )
            continue

        class_id = None

        try:
            class_id = int(parts[0])
        except Exception:
            problems.append(
                f"INVALID_CLASS line={line_number}"
            )
            continue

        values = [safe_float(x) for x in parts[1:]]

        if any(v is None or not math.isfinite(v) for v in values):
            problems.append(
                f"NON_NUMERIC line={line_number}"
            )
            continue

        xc, yc, bw, bh = values

        # ----------------------------------------------------
        # Class check
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:
            problems.append(
                f"UNKNOWN_CLASS line={line_number} class={class_id}"
            )

        # ----------------------------------------------------
        # YOLO coordinate checks
        # ----------------------------------------------------

        if not (0.0 <= xc <= 1.0):
            problems.append(
                f"XC_OUT_OF_RANGE line={line_number} xc={xc}"
            )

        if not (0.0 <= yc <= 1.0):
            problems.append(
                f"YC_OUT_OF_RANGE line={line_number} yc={yc}"
            )

        if bw <= 0 or bh <= 0:
            problems.append(
                f"NON_POSITIVE_SIZE line={line_number} w={bw} h={bh}"
            )
            continue

        if bw > 1 or bh > 1:
            problems.append(
                f"SIZE_OUT_OF_RANGE line={line_number} w={bw} h={bh}"
            )

        # ----------------------------------------------------
        # Convert to corner coordinates
        # ----------------------------------------------------

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        # ----------------------------------------------------
        # Check whether box extends outside image
        # ----------------------------------------------------

        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            problems.append(
                f"BOX_OUTSIDE_IMAGE line={line_number} "
                f"x1={x1:.6f} y1={y1:.6f} "
                f"x2={x2:.6f} y2={y2:.6f}"
            )

        # ----------------------------------------------------
        # Box area relative to image
        # ----------------------------------------------------

        box_area = bw * bh

        if box_area < VERY_SMALL_BOX_AREA:
            problems.append(
                f"VERY_SMALL_BOX line={line_number} "
                f"area={box_area:.6f}"
            )

        if box_area > VERY_LARGE_BOX_AREA:
            problems.append(
                f"VERY_LARGE_BOX line={line_number} "
                f"area={box_area:.6f}"
            )

        if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
            problems.append(
                f"EXTREMELY_THIN_BOX line={line_number} "
                f"w={bw:.6f} h={bh:.6f}"
            )

        # ----------------------------------------------------
        # Border contact
        # ----------------------------------------------------

        touches_border = (
            x1 <= BORDER_EPS
            or y1 <= BORDER_EPS
            or x2 >= 1 - BORDER_EPS
            or y2 >= 1 - BORDER_EPS
        )

        # Border contact is a FLAG, not automatically an error.
        if touches_border:
            problems.append(
                f"BORDER_TOUCH line={line_number}"
            )

        rows.append(
            {
                "line": line_number,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id, f"UNKNOWN_{class_id}"
                ),
                "xc": xc,
                "yc": yc,
                "width": bw,
                "height": bh,
                "area": box_area,
                "touches_border": touches_border,
                "image_width": width,
                "image_height": height,
            }
        )

    return rows, problems


def group_key(image_path):
    """
    Groups augmentation variants together.

    Examples:
        200_105.jpg
        200_105_45.jpg
        200_105_315.jpg
        200_105_b.jpg

    all become:

        200_105
    """

    name = image_path.stem

    for suffix in ["_45", "_315", "_b"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name


def main():

    print("=" * 70)
    print("Indian Currency YOLO Annotation Audit")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0

    class_images = Counter()
    class_boxes = Counter()

    all_problems = []

    split_stats = {}

    group_to_splits = defaultdict(set)
    group_to_images = defaultdict(list)

    area_values = []

    # ========================================================
    # PROCESS EACH SPLIT
    # ========================================================

    for split in SPLITS:

        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        print("\n" + "-" * 70)
        print(f"SCANNING: {split.upper()}")
        print("-" * 70)

        if not image_dir.exists():
            print(f"WARNING: Missing image directory: {image_dir}")
            continue

        if not label_dir.exists():
            print(f"WARNING: Missing label directory: {label_dir}")
            continue

        images = sorted(
            [
                p for p in image_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        split_images = 0
        split_boxes = 0
        split_problems = 0

        for index, image_path in enumerate(images, start=1):

            total_images += 1
            split_images += 1

            label_path = label_dir / f"{image_path.stem}.txt"

            # ------------------------------------------------
            # Missing label
            # ------------------------------------------------

            if not label_path.exists():

                message = (
                    f"{split} | {image_path.name} | "
                    f"MISSING_LABEL"
                )

                all_problems.append(message)
                split_problems += 1

                continue

            rows, problems = audit_label(
                label_path,
                image_path
            )

            # ------------------------------------------------
            # Group tracking
            # ------------------------------------------------

            group = group_key(image_path)

            group_to_splits[group].add(split)
            group_to_images[group].append(
                (split, image_path.name)
            )

            # ------------------------------------------------
            # Box statistics
            # ------------------------------------------------

            for row in rows:

                total_boxes += 1
                split_boxes += 1

                class_id = row["class_id"]

                class_images[class_id] += 1
                class_boxes[class_id] += 1

                area_values.append(row["area"])

            # ------------------------------------------------
            # Problems
            # ------------------------------------------------

            for problem in problems:

                message = (
                    f"{split} | {image_path.name} | {problem}"
                )

                all_problems.append(message)
                split_problems += 1

            if index % 500 == 0:
                print(
                    f"  Processed {index}/{len(images)} images..."
                )

        split_stats[split] = {
            "images": split_images,
            "boxes": split_boxes,
            "problems": split_problems,
        }

        print(f"\n{split}:")
        print(f"  Images  : {split_images}")
        print(f"  Boxes   : {split_boxes}")
        print(f"  Problems: {split_problems}")

    # ========================================================
    # GROUP LEAKAGE CHECK
    # ========================================================

    leakage_groups = {
        group: splits
        for group, splits in group_to_splits.items()
        if len(splits) > 1
    }

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for class_id in sorted(CLASS_NAMES):

        print(
            f"class {class_id} (₹{CLASS_NAMES[class_id]}): "
            f"{class_images[class_id]} boxes"
        )

    # ========================================================
    # BOX AREA STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("BOUNDING BOX AREA STATISTICS")
    print("=" * 70)

    if area_values:

        area_values.sort()

        def percentile(values, p):
            index = int((len(values) - 1) * p)
            return values[index]

        print(
            f"Minimum : {min(area_values):.6f}"
        )
        print(
            f"P05     : {percentile(area_values, 0.05):.6f}"
        )
        print(
            f"Median  : {percentile(area_values, 0.50):.6f}"
        )
        print(
            f"P95     : {percentile(area_values, 0.95):.6f}"
        )
        print(
            f"Maximum : {max(area_values):.6f}"
        )

    # ========================================================
    # LEAKAGE REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("GROUP LEAKAGE CHECK")
    print("=" * 70)

    if leakage_groups:

        print(
            f"WARNING: {len(leakage_groups)} groups "
            f"appear in multiple splits."
        )

        for group, splits in list(
            leakage_groups.items()
        )[:20]:

            print(
                f"  {group}: "
                f"{', '.join(sorted(splits))}"
            )

    else:

        print(
            "PASS: No augmentation group appears "
            "in multiple splits."
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = OUTPUT_DIR / "annotation_audit_report.txt"

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Indian Currency YOLO Annotation Audit\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Dataset: {DATASET}\n\n"
        )

        f.write(
            "SUMMARY\n"
        )
        f.write("-" * 70 + "\n")

        f.write(
            f"Total images: {total_images}\n"
        )

        f.write(
            f"Total boxes: {total_boxes}\n"
        )

        f.write(
            f"Total problems: {len(all_problems)}\n"
        )

        f.write(
            f"Groups appearing in multiple splits: "
            f"{len(leakage_groups)}\n\n"
        )

        f.write(
            "SPLIT STATISTICS\n"
        )
        f.write("-" * 70 + "\n")

        for split, stats in split_stats.items():

            f.write(
                f"{split}: "
                f"images={stats['images']}, "
                f"boxes={stats['boxes']}, "
                f"problems={stats['problems']}\n"
            )

        f.write("\n")

        f.write(
            "CLASS DISTRIBUTION\n"
        )
        f.write("-" * 70 + "\n")

        for class_id in sorted(CLASS_NAMES):

            f.write(
                f"class {class_id} "
                f"(₹{CLASS_NAMES[class_id]}): "
                f"{class_boxes[class_id]} boxes\n"
            )

        f.write("\n")

        f.write(
            "GROUP LEAKAGE\n"
        )
        f.write("-" * 70 + "\n")

        if leakage_groups:

            for group, splits in sorted(
                leakage_groups.items()
            ):

                f.write(
                    f"{group}: "
                    f"{', '.join(sorted(splits))}\n"
                )

        else:

            f.write(
                "NONE\n"
            )

        f.write("\n")

        f.write(
            "PROBLEMS\n"
        )
        f.write("-" * 70 + "\n")

        if all_problems:

            for problem in all_problems:
                f.write(problem + "\n")

        else:

            f.write(
                "NONE\n"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"\nImages checked : {total_images}")
    print(f"Boxes checked  : {total_boxes}")
    print(f"Problems found : {len(all_problems)}")

    print(
        f"\nReport:"
    )
    print(report_path)

    print("\nIMPORTANT:")
    print(
        "Problems are FLAGS for investigation."
    )
    print(
        "They are NOT automatically annotation errors."
    )
    print(
        "Do not modify the dataset based on this report yet."
    )


if __name__ == "__main__":
    main()