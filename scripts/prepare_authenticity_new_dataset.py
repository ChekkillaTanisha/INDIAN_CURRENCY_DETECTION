from pathlib import Path
import random
import shutil


# ============================================================
# PREPARE NEW AUTHENTICITY DATASET
#
# Source:
#   dataset/new_dataset/
#
# Output:
#   dataset/authenticity_new/
#
# Split:
#   70% train
#   15% validation
#   15% test
#
# REAL/FAKE and denomination folders are preserved.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "dataset" / "new_dataset"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "authenticity_new"


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = [
    "real",
    "fake",
]

DENOMINATIONS = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".avif",
    ".webp",
}


# ------------------------------------------------------------
# CHECK RATIOS
# ------------------------------------------------------------

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-9


# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

random.seed(SEED)


# ------------------------------------------------------------
# PRINT HEADER
# ------------------------------------------------------------

print("=" * 70)
print("PREPARING NEW AUTHENTICITY DATASET")
print("=" * 70)

print()

print(f"Source : {SOURCE_DIR}")
print(f"Output : {OUTPUT_DIR}")

print()

print("Split:")
print(f"  Train      : {TRAIN_RATIO * 100:.0f}%")
print(f"  Validation : {VAL_RATIO * 100:.0f}%")
print(f"  Test       : {TEST_RATIO * 100:.0f}%")

print()

# ------------------------------------------------------------
# CHECK SOURCE
# ------------------------------------------------------------

if not SOURCE_DIR.exists():

    raise FileNotFoundError(
        f"Source dataset does not exist:\n{SOURCE_DIR}"
    )


# ------------------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# ------------------------------------------------------------

for split in ["train", "validation", "test"]:

    for class_name in CLASSES:

        for denomination in DENOMINATIONS:

            output_path = (
                OUTPUT_DIR
                / split
                / class_name
                / denomination
            )

            output_path.mkdir(
                parents=True,
                exist_ok=True
            )


# ------------------------------------------------------------
# REMOVE OLD PREPARED DATASET
# ------------------------------------------------------------
#
# IMPORTANT:
# We only remove authenticity_new.
# We DO NOT touch new_dataset.
# We DO NOT touch authenticity_v3.
# ------------------------------------------------------------

print("Cleaning previous authenticity_new dataset...")

if OUTPUT_DIR.exists():

    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "validation", "test"]:

    for class_name in CLASSES:

        for denomination in DENOMINATIONS:

            (
                OUTPUT_DIR
                / split
                / class_name
                / denomination
            ).mkdir(
                parents=True,
                exist_ok=True
            )

print("Done.")

print()


# ------------------------------------------------------------
# COUNTERS
# ------------------------------------------------------------

counts = {

    split: {

        class_name: {

            denomination: 0

            for denomination in DENOMINATIONS

        }

        for class_name in CLASSES

    }

    for split in [
        "train",
        "validation",
        "test",
    ]

}


# ------------------------------------------------------------
# PROCESS EACH CLASS / DENOMINATION
# ------------------------------------------------------------

for class_name in CLASSES:

    for denomination in DENOMINATIONS:

        source_path = (
            SOURCE_DIR
            / class_name
            / denomination
        )

        print(
            f"Processing "
            f"{class_name.upper():<5} "
            f"₹{denomination:>4}..."
        )

        if not source_path.exists():

            print(
                f"  WARNING: Missing folder: "
                f"{source_path}"
            )

            continue


        # ----------------------------------------------------
        # COLLECT VALID IMAGE FILES
        # ----------------------------------------------------

        images = [

            path

            for path in source_path.iterdir()

            if (

                path.is_file()

                and

                path.suffix.lower()
                in VALID_EXTENSIONS

            )

        ]


        # ----------------------------------------------------
        # SORT FIRST FOR REPRODUCIBILITY
        # ----------------------------------------------------

        images = sorted(
            images,
            key=lambda p: p.name.lower()
        )


        print(
            f"  Found: {len(images)} images"
        )


        if len(images) == 0:

            continue


        # ----------------------------------------------------
        # SHUFFLE
        # ----------------------------------------------------

        random.shuffle(images)


        # ----------------------------------------------------
        # CALCULATE SPLIT SIZES
        # ----------------------------------------------------

        total = len(images)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        # Everything remaining goes to test.
        test_count = (
            total
            -
            train_count
            -
            val_count
        )


        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        train_images = (
            images[
                :train_count
            ]
        )

        validation_images = (
            images[
                train_count:
                train_count + val_count
            ]
        )

        test_images = (
            images[
                train_count + val_count:
            ]
        )


        split_data = {

            "train": train_images,

            "validation":
                validation_images,

            "test":
                test_images,

        }


        # ----------------------------------------------------
        # COPY FILES
        # ----------------------------------------------------

        for split, split_images in split_data.items():

            destination_dir = (
                OUTPUT_DIR
                / split
                / class_name
                / denomination
            )


            for image_path in split_images:

                destination_path = (
                    destination_dir
                    /
                    image_path.name
                )


                shutil.copy2(
                    image_path,
                    destination_path
                )


                counts[
                    split
                ][
                    class_name
                ][
                    denomination
                ] += 1


        print(
            f"  Train: {len(train_images)} | "
            f"Val: {len(validation_images)} | "
            f"Test: {len(test_images)}"
        )


print()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print()


grand_total = 0


for split in [
    "train",
    "validation",
    "test",
]:

    print(
        f"{split.upper()}:"
    )

    split_total = 0


    for class_name in CLASSES:

        class_total = 0


        for denomination in DENOMINATIONS:

            count = counts[
                split
            ][
                class_name
            ][
                denomination
            ]

            class_total += count


            print(
                f"  {class_name.upper():<5} "
                f"₹{denomination:>4}: "
                f"{count}"
            )


        print(
            f"  {class_name.upper():<5} TOTAL: "
            f"{class_total}"
        )


        split_total += class_total


    print(
        f"  SPLIT TOTAL: "
        f"{split_total}"
    )

    print()


    grand_total += split_total


print(
    f"TOTAL IMAGES: {grand_total}"
)

print()

print(
    "Dataset created at:"
)

print(
    OUTPUT_DIR.resolve()
)

print()

print(
    "Original dataset was NOT modified:"
)

print(
    SOURCE_DIR.resolve()
)

print()

print(
    "Old authenticity_v3 dataset was NOT modified."
)

print()

print("=" * 70)