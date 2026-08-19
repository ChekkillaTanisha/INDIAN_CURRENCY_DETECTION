from pathlib import Path
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# Indian Currency MobileNetV2 Classification
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset" / "mobilenet"
RUNS_DIR = PROJECT_ROOT / "runs" / "mobilenetv2"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RUNS_DIR / "best_mobilenetv2.pt"

DEVICE = torch.device("cpu")

NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0003
NUM_WORKERS = 0

CLASS_NAMES = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
]


print("=" * 70)
print("Indian Currency MobileNetV2 Training")
print("=" * 70)
print()

print(f"Dataset : {DATA_DIR}")
print(f"Output  : {RUNS_DIR}")
print(f"Device  : {DEVICE}")
print()


# ============================================================
# Verify dataset
# ============================================================

for split in ["train", "val", "test"]:
    split_dir = DATA_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing dataset split:\n{split_dir}"
        )


# ============================================================
# Image transformations
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transforms
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transforms
)


print("Class mapping:")
for name, index in train_dataset.class_to_idx.items():
    print(f"  {index}: ₹{name}")

print()

print(f"Train images: {len(train_dataset)}")
print(f"Val images  : {len(val_dataset)}")
print(f"Test images : {len(test_dataset)}")
print()


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MobileNetV2
# ============================================================

print("Loading pretrained MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Freeze feature extractor initially
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

print("MobileNetV2 loaded.")
print()


# ============================================================
# Loss / optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training functions
# ============================================================

def run_epoch(loader, training=False):

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if training:
                loss.backward()
                optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0
best_state = None

start_time = time.time()

print("=" * 70)
print("TRAINING")
print("=" * 70)
print()

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}%"
    )

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": CLASS_NAMES,
                "class_to_idx": train_dataset.class_to_idx,
                "val_accuracy": best_val_accuracy,
            },
            MODEL_PATH
        )

        print(
            f"  -> BEST MODEL SAVED "
            f"({best_val_accuracy * 100:.2f}%)"
        )

    print()

elapsed = time.time() - start_time


# ============================================================
# Restore best model
# ============================================================

if best_state is not None:
    model.load_state_dict(best_state)


# ============================================================
# Test evaluation
# ============================================================

print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)
print()

test_loss, test_accuracy = run_epoch(
    test_loader,
    training=False
)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print()


# ============================================================
# Final information
# ============================================================

print("=" * 70)
print("MOBILENETV2 TRAINING COMPLETE")
print("=" * 70)
print()

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Training time: "
    f"{elapsed / 60:.1f} minutes"
)

print()

print("Best model:")
print(MODEL_PATH)

print()
print("Class order:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: ₹{name}")

print()
print("=" * 70)