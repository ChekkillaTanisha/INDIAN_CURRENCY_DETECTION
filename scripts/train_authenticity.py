from pathlib import Path
import copy
import csv
import random
import time

import torch
import torch.nn as nn

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms


# ============================================================
# INDIAN CURRENCY AUTHENTICITY - MOBILENETV2
#
# FAKE vs REAL
#
# NEW DATASET VERSION
#
# Dataset:
#
# dataset/
# └── authenticity_new/
#     ├── train/
#     │   ├── fake/
#     │   │   ├── 10/
#     │   │   ├── 20/
#     │   │   ├── 50/
#     │   │   ├── 100/
#     │   │   ├── 200/
#     │   │   ├── 500/
#     │   │   └── 2000/
#     │   │
#     │   └── real/
#     │       ├── 10/
#     │       ├── 20/
#     │       ├── 50/
#     │       ├── 100/
#     │       ├── 200/
#     │       ├── 500/
#     │       └── 2000/
#     │
#     ├── validation/
#     │   ├── fake/
#     │   └── real/
#     │
#     └── test/
#         ├── fake/
#         └── real/
#
# Total:
#   7,445 images
#
# Training:
#   5,206 images
#
# Validation:
#   1,110 images
#
# Test:
#   1,129 images
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "authenticity_new"
)

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"


RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_new"
)

RUNS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    RUNS_DIR
    / "best_authenticity_mobilenetv2_new.pt"
)

HISTORY_PATH = (
    RUNS_DIR
    / "training_history.csv"
)

ERRORS_PATH = (
    RUNS_DIR
    / "test_errors.csv"
)

PREDICTIONS_PATH = (
    RUNS_DIR
    / "test_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# CPU because your current setup is using CPU.
DEVICE = torch.device("cpu")

NUM_CLASSES = 2

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
    ".avif",
}

SEED = 42

BATCH_SIZE = 32

# 15 epochs is a good time-conscious setting.
# Early stopping can stop before all 15 epochs.
EPOCHS = 15

LEARNING_RATE = 0.0001

NUM_WORKERS = 0

WEIGHT_DECAY = 1e-4

# Stop if validation Macro-F1 does not improve
# for 4 consecutive epochs.
PATIENCE = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 70)
print("INDIAN CURRENCY AUTHENTICITY - MobileNetV2")
print("NEW DATASET TRAINING")
print("=" * 70)

print()

print(f"Dataset : {DATA_DIR}")
print(f"Output  : {RUNS_DIR}")
print(f"Device  : {DEVICE}")

print()


# ============================================================
# DATASET
# ============================================================

class AuthenticityDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
    ):

        self.root_dir = Path(root_dir)

        self.transform = transform

        self.samples = []

        # ----------------------------------------------------
        # FAKE = 0
        # REAL = 1
        # ----------------------------------------------------

        class_directories = {
            "fake": 0,
            "real": 1,
        }

        for class_name, label in class_directories.items():

            class_dir = (
                self.root_dir
                / class_name
            )

            if not class_dir.exists():

                print(
                    f"WARNING: Missing directory: "
                    f"{class_dir}"
                )

                continue

            # Recursively find images.
            # This handles:
            #
            # train/fake/10/image.jpg
            # train/fake/20/image.jpg
            #
            # as well as images directly inside fake/.
            for image_path in class_dir.rglob("*"):

                if (
                    image_path.is_file()
                    and
                    image_path.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):

                    self.samples.append(
                        {
                            "path": image_path,
                            "label": label,
                        }
                    )

        # Deterministic ordering
        self.samples.sort(
            key=lambda x: str(x["path"]).lower()
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[index]

        image_path = sample["path"]

        label = sample["label"]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

# Training augmentation.
#
# These transformations help the model handle:
# - different lighting
# - small rotations
# - different camera positions
# - small scale differences
# - real-world image conditions
#
# We do NOT use aggressive transformations because
# currency security features should not be destroyed.

train_transform = transforms.Compose(
    [

        transforms.Resize(
            (256, 256)
        ),

        transforms.RandomResizedCrop(
            224,
            scale=(0.85, 1.0),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),

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
    ]
)


# Validation/test transformation.
eval_transform = transforms.Compose(
    [

        transforms.Resize(
            (256, 256)
        ),

        transforms.CenterCrop(
            224
        ),

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
    ]
)


# ============================================================
# CREATE DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

print()


train_dataset = AuthenticityDataset(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = AuthenticityDataset(
    VAL_DIR,
    transform=eval_transform,
)

test_dataset = AuthenticityDataset(
    TEST_DIR,
    transform=eval_transform,
)


print(
    f"Training images   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation images : "
    f"{len(val_dataset)}"
)

print(
    f"Test images       : "
    f"{len(test_dataset)}"
)

print()


# ============================================================
# CHECK DATASET
# ============================================================

if len(train_dataset) == 0:

    raise RuntimeError(
        f"No training images found in:\n"
        f"{TRAIN_DIR}"
    )


if len(val_dataset) == 0:

    raise RuntimeError(
        f"No validation images found in:\n"
        f"{VAL_DIR}"
    )


if len(test_dataset) == 0:

    raise RuntimeError(
        f"No test images found in:\n"
        f"{TEST_DIR}"
    )


# ============================================================
# DATASET CLASS COUNTS
# ============================================================

def count_classes(dataset):

    fake_count = 0

    real_count = 0

    for sample in dataset.samples:

        if sample["label"] == 0:

            fake_count += 1

        else:

            real_count += 1

    return fake_count, real_count


train_fake, train_real = count_classes(
    train_dataset
)

val_fake, val_real = count_classes(
    val_dataset
)

test_fake, test_real = count_classes(
    test_dataset
)


print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print()

print(
    f"TRAIN:"
)

print(
    f"  FAKE : {train_fake}"
)

print(
    f"  REAL : {train_real}"
)

print()

print(
    f"VALIDATION:"
)

print(
    f"  FAKE : {val_fake}"
)

print(
    f"  REAL : {val_real}"
)

print()

print(
    f"TEST:"
)

print(
    f"  FAKE : {test_fake}"
)

print(
    f"  REAL : {test_real}"
)

print()


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("CREATING MobileNetV2")
print("=" * 70)

print()


# Use ImageNet pretrained MobileNetV2.
#
# This gives the network useful visual features before
# training on Indian currency.
#
# If your installed torchvision is older and does not support
# weights=..., change this to:
#
# models.mobilenet_v2(pretrained=True)

try:

    weights = (
        models.MobileNet_V2_Weights.DEFAULT
    )

    model = models.mobilenet_v2(
        weights=weights
    )

except AttributeError:

    model = models.mobilenet_v2(
        pretrained=True
    )


# Replace final classifier.
#
# Original MobileNetV2 output:
# 1000 ImageNet classes
#
# New output:
# 2 classes
#
# 0 = FAKE
# 1 = REAL

in_features = (
    model.classifier[-1].in_features
)


model.classifier[-1] = nn.Linear(
    in_features,
    NUM_CLASSES,
)


model = model.to(
    DEVICE
)


print(
    "Architecture : MobileNetV2"
)

print(
    "Classes      : FAKE / REAL"
)

print()


# ============================================================
# CLASS WEIGHTS
# ============================================================
#
# REAL has more training images than FAKE.
#
# Weighted loss prevents the model from simply favoring REAL.
#
# Weight is inversely related to class frequency.
# ============================================================

total_train = (
    train_fake
    +
    train_real
)

fake_weight = (
    total_train
    /
    (2.0 * train_fake)
)

real_weight = (
    total_train
    /
    (2.0 * train_real)
)


class_weights = torch.tensor(
    [
        fake_weight,
        real_weight,
    ],
    dtype=torch.float32,
    device=DEVICE,
)


print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

print()

print(
    f"FAKE weight : {fake_weight:.4f}"
)

print(
    f"REAL weight : {real_weight:.4f}"
)

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0

    correct = 0

    total_count = 0


    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        batch_size = (
            labels.size(0)
        )


        running_loss += (
            loss.item()
            *
            batch_size
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        correct += (
            predictions
            ==
            labels
        ).sum().item()


        total_count += batch_size


    epoch_loss = (
        running_loss
        /
        total_count
    )


    epoch_accuracy = (
        correct
        /
        total_count
    )


    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total_count = 0


    confusion = {

        "fake_fake": 0,

        "fake_real": 0,

        "real_fake": 0,

        "real_real": 0,
    }


    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        batch_size = (
            labels.size(0)
        )


        running_loss += (
            loss.item()
            *
            batch_size
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        correct += (
            predictions
            ==
            labels
        ).sum().item()


        total_count += batch_size


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        for true_label, predicted_label in zip(
            labels.cpu().tolist(),
            predictions.cpu().tolist(),
        ):

            if (
                true_label == 0
                and predicted_label == 0
            ):

                confusion["fake_fake"] += 1


            elif (
                true_label == 0
                and predicted_label == 1
            ):

                confusion["fake_real"] += 1


            elif (
                true_label == 1
                and predicted_label == 0
            ):

                confusion["real_fake"] += 1


            elif (
                true_label == 1
                and predicted_label == 1
            ):

                confusion["real_real"] += 1


    epoch_loss = (
        running_loss
        /
        total_count
    )


    epoch_accuracy = (
        correct
        /
        total_count
    )


    return (
        epoch_loss,
        epoch_accuracy,
        confusion,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    confusion,
):

    tp_fake = confusion[
        "fake_fake"
    ]

    fn_fake = confusion[
        "fake_real"
    ]

    fp_fake = confusion[
        "real_fake"
    ]

    tn_fake = confusion[
        "real_real"
    ]


    # --------------------------------------------------------
    # FAKE
    # --------------------------------------------------------

    fake_precision = (

        tp_fake
        /
        (tp_fake + fp_fake)

        if (
            tp_fake + fp_fake
        ) > 0

        else 0.0
    )


    fake_recall = (

        tp_fake
        /
        (tp_fake + fn_fake)

        if (
            tp_fake + fn_fake
        ) > 0

        else 0.0
    )


    fake_f1 = (

        2
        *
        fake_precision
        *
        fake_recall
        /
        (
            fake_precision
            +
            fake_recall
        )

        if (
            fake_precision
            +
            fake_recall
        ) > 0

        else 0.0
    )


    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    real_precision = (

        tn_fake
        /
        (tn_fake + fn_fake)

        if (
            tn_fake + fn_fake
        ) > 0

        else 0.0
    )


    real_recall = (

        tn_fake
        /
        (tn_fake + fp_fake)

        if (
            tn_fake + fp_fake
        ) > 0

        else 0.0
    )


    real_f1 = (

        2
        *
        real_precision
        *
        real_recall
        /
        (
            real_precision
            +
            real_recall
        )

        if (
            real_precision
            +
            real_recall
        ) > 0

        else 0.0
    )


    macro_f1 = (
        fake_f1
        +
        real_f1
    ) / 2.0


    return {

        "fake_precision":
            fake_precision,

        "fake_recall":
            fake_recall,

        "fake_f1":
            fake_f1,

        "real_precision":
            real_precision,

        "real_recall":
            real_recall,

        "real_f1":
            real_f1,

        "macro_f1":
            macro_f1,
    }


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()

print(
    f"Epochs       : {EPOCHS}"
)

print(
    f"Batch size   : {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Device       : {DEVICE}"
)

print()


best_val_f1 = -1.0

best_val_accuracy = 0.0

best_model_state = None

epochs_without_improvement = 0

training_history = []


training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = time.time()


    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
    )


    val_loss, val_accuracy, val_confusion = (
        evaluate(
            model,
            val_loader,
            criterion,
        )
    )


    val_metrics = calculate_metrics(
        val_confusion
    )


    val_f1 = val_metrics[
        "macro_f1"
    ]


    scheduler.step(
        val_f1
    )


    epoch_time = (
        time.time()
        -
        epoch_start
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"Val Macro-F1: {val_f1 * 100:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.1f}s"
    )


    training_history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "val_macro_f1":
                val_f1,

            "learning_rate":
                current_lr,
        }
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_val_accuracy = (
            val_accuracy
        )


        best_model_state = (
            copy.deepcopy(
                model.state_dict()
            )
        )


        epochs_without_improvement = 0


        checkpoint = {

            "model_state_dict":
                best_model_state,

            "class_names":
                CLASS_NAMES,

            "num_classes":
                NUM_CLASSES,

            "best_val_accuracy":
                best_val_accuracy,

            "best_val_macro_f1":
                best_val_f1,

            "architecture":
                "mobilenet_v2",

            "dataset":
                "authenticity_new",

            "dataset_format":
                "train_validation_test_nested_fake_real",

            "seed":
                SEED,
        }


        torch.save(
            checkpoint,
            MODEL_PATH,
        )


        print(
            f"  -> Best model saved "
            f"(Val Macro-F1: "
            f"{best_val_f1 * 100:.2f}%)"
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            f"Early stopping after "
            f"{epoch} epochs."
        )

        break


print()


training_time = (
    time.time()
    -
    training_start
)


print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)

print()


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "learning_rate",
        ],
    )


    writer.writeheader()


    for row in training_history:

        writer.writerow(
            row
        )


print(
    f"Training history saved:\n"
    f"{HISTORY_PATH}"
)

print()


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_model_state is None:

    raise RuntimeError(
        "No best model was produced."
    )


model.load_state_dict(
    best_model_state
)


model.eval()


# ============================================================
# FINAL VALIDATION
# ============================================================

print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print()


val_loss, val_accuracy, val_confusion = (
    evaluate(
        model,
        val_loader,
        criterion,
    )
)


val_metrics = calculate_metrics(
    val_confusion
)


print(
    f"Validation loss     : "
    f"{val_loss:.4f}"
)

print(
    f"Validation accuracy : "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro-F1 : "
    f"{val_metrics['macro_f1'] * 100:.2f}%"
)

print()


# ============================================================
# FINAL TEST
#
# The test dataset has NEVER been used for training.
# ============================================================

print("=" * 70)
print("FINAL TEST")
print("=" * 70)

print()


test_loss, test_accuracy, test_confusion = (
    evaluate(
        model,
        test_loader,
        criterion,
    )
)


test_metrics = calculate_metrics(
    test_confusion
)


print(
    f"Test loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro-F1 : "
    f"{test_metrics['macro_f1'] * 100:.2f}%"
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("TEST CONFUSION MATRIX")
print("=" * 70)

print()


print(
    f"FAKE predicted FAKE : "
    f"{test_confusion['fake_fake']}"
)

print(
    f"FAKE predicted REAL : "
    f"{test_confusion['fake_real']}"
)

print(
    f"REAL predicted FAKE : "
    f"{test_confusion['real_fake']}"
)

print(
    f"REAL predicted REAL : "
    f"{test_confusion['real_real']}"
)

print()


# ============================================================
# CLASS PERFORMANCE
# ============================================================

print("=" * 70)
print("CLASS PERFORMANCE")
print("=" * 70)

print()


print(
    f"FAKE precision : "
    f"{test_metrics['fake_precision'] * 100:.2f}%"
)

print(
    f"FAKE recall    : "
    f"{test_metrics['fake_recall'] * 100:.2f}%"
)

print(
    f"FAKE F1        : "
    f"{test_metrics['fake_f1'] * 100:.2f}%"
)

print()


print(
    f"REAL precision : "
    f"{test_metrics['real_precision'] * 100:.2f}%"
)

print(
    f"REAL recall    : "
    f"{test_metrics['real_recall'] * 100:.2f}%"
)

print(
    f"REAL F1        : "
    f"{test_metrics['real_f1'] * 100:.2f}%"
)

print()


# ============================================================
# TEST PREDICTIONS + ERROR ANALYSIS
# ============================================================

print("=" * 70)
print("TEST ERROR ANALYSIS")
print("=" * 70)

print()


errors = []

all_predictions = []


@torch.no_grad()
def collect_test_predictions(
    model,
    samples,
):

    model.eval()


    for sample in samples:

        image_path = sample["path"]

        true_label = sample["label"]


        try:

            image = Image.open(
                image_path
            ).convert("RGB")


            image_tensor = (
                eval_transform(
                    image
                )
                .unsqueeze(0)
                .to(DEVICE)
            )


            outputs = model(
                image_tensor
            )


            probabilities = torch.softmax(
                outputs,
                dim=1,
            )


            predicted_label = (
                probabilities
                .argmax(
                    dim=1
                )
                .item()
            )


            fake_probability = (
                probabilities[
                    0,
                    0,
                ].item()
            )


            real_probability = (
                probabilities[
                    0,
                    1,
                ].item()
            )


            confidence = (
                probabilities[
                    0,
                    predicted_label,
                ].item()
            )


            prediction = {

                "image":
                    str(image_path),

                "true_label":
                    CLASS_NAMES[
                        true_label
                    ],

                "predicted_label":
                    CLASS_NAMES[
                        predicted_label
                    ],

                "fake_probability":
                    fake_probability,

                "real_probability":
                    real_probability,

                "confidence":
                    confidence,
            }


            all_predictions.append(
                prediction
            )


            if (
                predicted_label
                !=
                true_label
            ):

                errors.append(
                    prediction
                )


        except Exception as e:

            print(
                f"Could not process: "
                f"{image_path}"
            )

            print(
                f"Error: {e}"
            )


collect_test_predictions(
    model,
    test_dataset.samples,
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

with open(
    PREDICTIONS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)


    writer.writerow(
        [
            "image",
            "true_label",
            "predicted_label",
            "fake_probability",
            "real_probability",
            "confidence",
        ]
    )


    for prediction in all_predictions:

        writer.writerow(
            [
                prediction["image"],
                prediction["true_label"],
                prediction["predicted_label"],
                f"{prediction['fake_probability']:.6f}",
                f"{prediction['real_probability']:.6f}",
                f"{prediction['confidence']:.6f}",
            ]
        )


print(
    f"Test predictions saved:\n"
    f"{PREDICTIONS_PATH}"
)

print()


# ============================================================
# SAVE ERRORS
# ============================================================

with open(
    ERRORS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)


    writer.writerow(
        [
            "image",
            "true_label",
            "predicted_label",
            "fake_probability",
            "real_probability",
            "confidence",
        ]
    )


    for error in errors:

        writer.writerow(
            [
                error["image"],
                error["true_label"],
                error["predicted_label"],
                f"{error['fake_probability']:.6f}",
                f"{error['real_probability']:.6f}",
                f"{error['confidence']:.6f}",
            ]
        )


print(
    f"Wrong test images: "
    f"{len(errors)}"
)

print(
    f"Error file saved:\n"
    f"{ERRORS_PATH}"
)

print()


# ============================================================
# FINAL MODEL CHECKPOINT
# ============================================================

final_checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "class_names":
        CLASS_NAMES,

    "num_classes":
        NUM_CLASSES,

    "best_val_accuracy":
        best_val_accuracy,

    "best_val_macro_f1":
        best_val_f1,

    "test_accuracy":
        test_accuracy,

    "test_macro_f1":
        test_metrics["macro_f1"],

    "fake_precision":
        test_metrics["fake_precision"],

    "fake_recall":
        test_metrics["fake_recall"],

    "fake_f1":
        test_metrics["fake_f1"],

    "real_precision":
        test_metrics["real_precision"],

    "real_recall":
        test_metrics["real_recall"],

    "real_f1":
        test_metrics["real_f1"],

    "architecture":
        "mobilenet_v2",

    "dataset":
        "authenticity_new",

    "dataset_format":
        "train_validation_test_nested_fake_real",

    "train_images":
        len(train_dataset),

    "validation_images":
        len(val_dataset),

    "test_images":
        len(test_dataset),

    "seed":
        SEED,

    "training_epochs":
        len(training_history),
}


torch.save(
    final_checkpoint,
    MODEL_PATH,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()


print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print()


print(
    f"Best validation accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best validation Macro-F1 : "
    f"{best_val_f1 * 100:.2f}%"
)

print()


print(
    f"Test accuracy            : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro-F1            : "
    f"{test_metrics['macro_f1'] * 100:.2f}%"
)

print()


print(
    f"FAKE precision           : "
    f"{test_metrics['fake_precision'] * 100:.2f}%"
)

print(
    f"FAKE recall              : "
    f"{test_metrics['fake_recall'] * 100:.2f}%"
)

print(
    f"FAKE F1                  : "
    f"{test_metrics['fake_f1'] * 100:.2f}%"
)

print()


print(
    f"REAL precision           : "
    f"{test_metrics['real_precision'] * 100:.2f}%"
)

print(
    f"REAL recall              : "
    f"{test_metrics['real_recall'] * 100:.2f}%"
)

print(
    f"REAL F1                  : "
    f"{test_metrics['real_f1'] * 100:.2f}%"
)

print()


print(
    "Next step:"
)

print(
    "Use this new MobileNetV2 authenticity "
    "model together with the existing YOLO11 "
    "denomination detector."
)

print()

print("=" * 70)