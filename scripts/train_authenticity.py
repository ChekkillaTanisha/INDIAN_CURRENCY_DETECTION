from pathlib import Path
from collections import defaultdict, Counter
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
# INDIAN CURRENCY AUTHENTICITY - MOBILENETV2 V3
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "authenticity"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v2"
)

MODEL_PATH = (
    RUNS_DIR
    / "best_authenticity_mobilenetv2_v2.pt"
)

TEST_SPLIT_PATH = (
    RUNS_DIR
    / "test_split.csv"
)

ERRORS_PATH = (
    RUNS_DIR
    / "test_errors.csv"
)


RUNS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BATCH_SIZE = 32

EPOCHS = 12

PATIENCE = 3

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

NUM_WORKERS = 0

DEVICE = torch.device("cpu")


# ============================================================
# DENOMINATIONS
#
# ₹2000 intentionally excluded.
# ============================================================

DENOMINATIONS = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 70)
print("Indian Currency Authenticity MobileNetV2 V3 Training")
print("=" * 70)

print()

print(f"Dataset : {DATA_DIR}")
print(f"Output  : {RUNS_DIR}")
print(f"Device  : {DEVICE}")

print()

print("Classes:")

print("  0: FAKE")
print("  1: REAL")

print()

print("Denominations:")

for denomination in DENOMINATIONS:
    print(f"  ₹{denomination}")

print()

print("₹2000 is EXCLUDED.")

print()


# ============================================================
# VERIFY DATASET
# ============================================================

if not DATA_DIR.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_DIR}"
    )


for authenticity in [
    "real",
    "fake",
]:

    folder = (
        DATA_DIR
        / authenticity
    )

    if not folder.exists():

        raise FileNotFoundError(
            f"Missing folder:\n{folder}"
        )


# ============================================================
# COLLECT IMAGES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


all_samples = []


for authenticity in [
    "fake",
    "real",
]:

    for denomination in DENOMINATIONS:

        folder = (
            DATA_DIR
            / authenticity
            / denomination
        )

        if not folder.exists():

            print(
                f"WARNING: Missing folder: {folder}"
            )

            continue


        for image_path in folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                label = (
                    0
                    if authenticity == "fake"
                    else 1
                )


                all_samples.append(
                    {
                        "path": image_path,
                        "label": label,
                        "authenticity": authenticity,
                        "denomination": denomination,
                    }
                )


print(
    f"Total images found: {len(all_samples)}"
)

print()


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

print("=" * 70)
print("DATASET DISTRIBUTION")
print("=" * 70)

print()


for denomination in DENOMINATIONS:

    real_count = sum(
        1
        for x in all_samples
        if (
            x["denomination"]
            == denomination
            and
            x["authenticity"]
            == "real"
        )
    )


    fake_count = sum(
        1
        for x in all_samples
        if (
            x["denomination"]
            == denomination
            and
            x["authenticity"]
            == "fake"
        )
    )


    print(
        f"₹{denomination:>4}: "
        f"REAL={real_count:<4} "
        f"FAKE={fake_count:<4} "
        f"TOTAL={real_count + fake_count}"
    )


print()


# ============================================================
# STRATIFIED SPLIT
#
# 70% train
# 15% validation
# 15% test
#
# Exact test filenames are saved below.
# ============================================================

groups = defaultdict(list)


for sample in all_samples:

    key = (
        sample["denomination"],
        sample["authenticity"],
    )

    groups[key].append(sample)


train_samples = []

val_samples = []

test_samples = []


for key in sorted(groups.keys()):

    samples = groups[key]

    # IMPORTANT:
    # deterministic shuffle
    random.shuffle(samples)

    n = len(samples)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )


    train_samples.extend(
        samples[
            :train_end
        ]
    )


    val_samples.extend(
        samples[
            train_end:val_end
        ]
    )


    test_samples.extend(
        samples[
            val_end:
        ]
    )


random.shuffle(train_samples)

random.shuffle(val_samples)

random.shuffle(test_samples)


print("=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print()

print(
    f"Train: {len(train_samples)}"
)

print(
    f"Val  : {len(val_samples)}"
)

print(
    f"Test : {len(test_samples)}"
)

print()

print(
    f"Total: "
    f"{len(train_samples) + len(val_samples) + len(test_samples)}"
)

print()


# ============================================================
# SAVE EXACT TEST SPLIT
# ============================================================

with open(
    TEST_SPLIT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "path",
            "label",
            "authenticity",
            "denomination",
        ]
    )


    for sample in test_samples:

        writer.writerow(
            [
                str(
                    sample["path"]
                ),
                sample["label"],
                sample["authenticity"],
                sample["denomination"],
            ]
        )


print(
    "Exact test split saved:"
)

print(
    TEST_SPLIT_PATH
)

print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transforms = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(
            0.03,
            0.03
        ),
        scale=(
            0.95,
            1.05
        ),
    ),

    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12,
        saturation=0.08,
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
])


eval_transforms = transforms.Compose([

    transforms.Resize(
        (224, 224)
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
])


# ============================================================
# DATASET
# ============================================================

class AuthenticityDataset(
    Dataset
):

    def __init__(
        self,
        samples,
        transform=None,
    ):

        self.samples = samples

        self.transform = transform


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        sample = (
            self.samples[index]
        )


        try:

            image = Image.open(
                sample["path"]
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not read:\n"
                f"{sample['path']}\n"
                f"{e}"
            )


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            sample["label"],
            sample["denomination"],
            str(sample["path"]),
        )


# ============================================================
# DATASETS
# ============================================================

train_dataset = AuthenticityDataset(
    train_samples,
    train_transforms,
)

val_dataset = AuthenticityDataset(
    val_samples,
    eval_transforms,
)

test_dataset = AuthenticityDataset(
    test_samples,
    eval_transforms,
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_label_counts = Counter(
    sample["label"]
    for sample in train_samples
)


fake_count = (
    train_label_counts[0]
)

real_count = (
    train_label_counts[1]
)

total_train = (
    fake_count
    + real_count
)


fake_weight = (
    total_train
    /
    (2.0 * fake_count)
)


real_weight = (
    total_train
    /
    (2.0 * real_count)
)


class_weights = torch.tensor(
    [
        fake_weight,
        real_weight,
    ],
    dtype=torch.float32,
).to(DEVICE)


print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

print()

print(
    f"FAKE count   : {fake_count}"
)

print(
    f"REAL count   : {real_count}"
)

print(
    f"FAKE weight  : {fake_weight:.4f}"
)

print(
    f"REAL weight  : {real_weight:.4f}"
)

print()


# ============================================================
# MOBILENETV2
# ============================================================

print("=" * 70)
print("LOADING PRETRAINED MOBILENETV2")
print("=" * 70)

print()


weights = (
    models.MobileNet_V2_Weights.DEFAULT
)


model = models.mobilenet_v2(
    weights=weights
)


# Freeze everything first

for parameter in (
    model.features.parameters()
):

    parameter.requires_grad = False


# Fine-tune later layers

for parameter in (
    model.features[14:].parameters()
):

    parameter.requires_grad = True


# Replace classifier

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2,
)


model = model.to(DEVICE)


print(
    "MobileNetV2 loaded."
)

print(
    "Fine-tuning later layers."
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

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]


optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
)


# ============================================================
# RUN EPOCH
# ============================================================

def run_epoch(
    loader,
    training=False,
):

    if training:

        model.train()

    else:

        model.eval()


    running_loss = 0.0

    correct = 0

    total = 0


    with torch.set_grad_enabled(
        training
    ):

        for (
            images,
            labels,
            _,
            _,
        ) in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            if training:

                optimizer.zero_grad()


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            if training:

                loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )


                optimizer.step()


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            correct += (
                predictions
                == labels
            ).sum().item()


            total += (
                labels.size(0)
            )


    return (
        running_loss / total,
        correct / total,
    )


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

best_state = None

epochs_without_improvement = 0

start_time = time.time()


print("=" * 70)
print("TRAINING V3")
print("=" * 70)

print()


for epoch in range(
    1,
    EPOCHS + 1,
):

    train_loss, train_acc = (
        run_epoch(
            train_loader,
            training=True,
        )
    )


    val_loss, val_acc = (
        run_epoch(
            val_loader,
            training=False,
        )
    )


    scheduler.step(
        val_acc
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}% | "
        f"LR: {current_lr:.6f}"
    )


    if (
        val_acc
        > best_val_accuracy
    ):

        best_val_accuracy = val_acc

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": [
                    "fake",
                    "real",
                ],
                "val_accuracy":
                    best_val_accuracy,
                "denominations":
                    DENOMINATIONS,
                "version":
                    "V3",
                "seed":
                    SEED,
            },
            MODEL_PATH,
        )


        print(
            f"  -> BEST MODEL SAVED "
            f"({best_val_accuracy * 100:.2f}%)"
        )


    else:

        epochs_without_improvement += 1


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            "Early stopping."
        )

        break


    print()


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# EXACT TEST EVALUATION
#
# This also records every wrong image.
# ============================================================

print("=" * 70)
print("EXACT TEST EVALUATION")
print("=" * 70)

print()


model.eval()


total = 0

correct = 0

wrong = 0


confusion = {
    "fake_fake": 0,
    "fake_real": 0,
    "real_fake": 0,
    "real_real": 0,
}


errors = []


denomination_results = defaultdict(
    lambda: {
        "correct": 0,
        "total": 0,
    }
)


with torch.no_grad():

    for (
        images,
        labels,
        denominations,
        paths,
    ) in test_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        outputs = model(
            images
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        for i in range(
            len(labels)
        ):

            actual = (
                labels[i].item()
            )

            predicted = (
                predictions[i].item()
            )

            confidence = float(
                probabilities[
                    i,
                    predicted
                ].item()
            )


            denomination = (
                denominations[i]
            )

            image_path = (
                paths[i]
            )


            total += 1


            denomination_results[
                denomination
            ]["total"] += 1


            if actual == predicted:

                correct += 1

                denomination_results[
                    denomination
                ]["correct"] += 1

            else:

                wrong += 1

                errors.append(
                    {
                        "image":
                            image_path,
                        "denomination":
                            denomination,
                        "actual":
                            "FAKE"
                            if actual == 0
                            else "REAL",
                        "predicted":
                            "FAKE"
                            if predicted == 0
                            else "REAL",
                        "confidence":
                            confidence,
                    }
                )


            if actual == 0 and predicted == 0:

                confusion[
                    "fake_fake"
                ] += 1

            elif actual == 0 and predicted == 1:

                confusion[
                    "fake_real"
                ] += 1

            elif actual == 1 and predicted == 0:

                confusion[
                    "real_fake"
                ] += 1

            elif actual == 1 and predicted == 1:

                confusion[
                    "real_real"
                ] += 1


# ============================================================
# SAVE ERROR REPORT
# ============================================================

with open(
    ERRORS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "denomination",
            "actual",
            "predicted",
            "confidence",
        ],
    )


    writer.writeheader()


    writer.writerows(
        errors
    )


# ============================================================
# RESULTS
# ============================================================

test_accuracy = (
    correct / total
    if total > 0
    else 0.0
)


elapsed = (
    time.time()
    - start_time
)


print()

print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print()

print(
    f"Test images : {total}"
)

print(
    f"Correct     : {correct}"
)

print(
    f"Wrong       : {wrong}"
)

print(
    f"Accuracy    : {test_accuracy * 100:.2f}%"
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("AUTHENTICITY CONFUSION MATRIX")
print("=" * 70)

print()

print(
    f"FAKE predicted FAKE : "
    f"{confusion['fake_fake']}"
)

print(
    f"FAKE predicted REAL : "
    f"{confusion['fake_real']}"
)

print(
    f"REAL predicted FAKE : "
    f"{confusion['real_fake']}"
)

print(
    f"REAL predicted REAL : "
    f"{confusion['real_real']}"
)

print()


# ============================================================
# PER DENOMINATION
# ============================================================

print("=" * 70)
print("PER-DENOMINATION RESULTS")
print("=" * 70)

print()


for denomination in DENOMINATIONS:

    result = (
        denomination_results[
            denomination
        ]
    )


    denom_total = (
        result["total"]
    )

    denom_correct = (
        result["correct"]
    )


    accuracy = (
        denom_correct
        / denom_total
        * 100
        if denom_total > 0
        else 0
    )


    print(
        f"₹{denomination:>4} : "
        f"{denom_correct}/{denom_total} "
        f"= {accuracy:.2f}%"
    )


print()


# ============================================================
# INCORRECT IMAGES
# ============================================================

print("=" * 70)
print("INCORRECT TEST IMAGES")
print("=" * 70)

print()


if len(errors) == 0:

    print(
        "No incorrect test images."
    )

else:

    for number, error in enumerate(
        errors,
        start=1,
    ):

        print(
            f"{number}. "
            f"{error['image']}"
        )

        print(
            f"   Denomination : "
            f"₹{error['denomination']}"
        )

        print(
            f"   Actual       : "
            f"{error['actual']}"
        )

        print(
            f"   Predicted    : "
            f"{error['predicted']}"
        )

        print(
            f"   Confidence   : "
            f"{error['confidence'] * 100:.2f}%"
        )

        print()


print(
    "Error report:"
)

print(
    ERRORS_PATH
)

print()


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("=" * 70)
print("AUTHENTICITY V3 TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Exact test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Wrong test images: "
    f"{wrong}"
)

print(
    f"Training time: "
    f"{elapsed / 60:.1f} minutes"
)

print()

print("Best model:")

print(
    MODEL_PATH
)

print()

print("Exact test split:")

print(
    TEST_SPLIT_PATH
)

print()

print("Error report:")

print(
    ERRORS_PATH
)

print()

print("Class order:")

print("  0: FAKE")

print("  1: REAL")

print()

print("₹2000 excluded.")

print()

print("=" * 70)