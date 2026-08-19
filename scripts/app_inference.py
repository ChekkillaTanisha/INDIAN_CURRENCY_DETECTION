from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# INDIAN CURRENCY DETECTION
# YOLO11 + MobileNetV2 V2
#
# YOLO11:
#   Detects the currency note and denomination.
#
# MobileNetV2:
#   Detects REAL vs FAKE.
#
# This file performs INFERENCE ONLY.
# It does NOT train the models.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "currency_yolo11n"
    / "weights"
    / "best.pt"
)


AUTHENTICITY_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_v2"
    / "best_authenticity_mobilenetv2_v2.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# YOLO DENOMINATION CLASSES
# ============================================================

YOLO_CLASS_NAMES = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
]


# ============================================================
# AUTHENTICITY CLASSES
#
# IMPORTANT:
#
# 0 = FAKE
# 1 = REAL
# ============================================================

AUTHENTICITY_CLASS_NAMES = [
    "FAKE",
    "REAL",
]


# ============================================================
# AUTHENTICITY SUPPORTED DENOMINATIONS
#
# ₹2000 was excluded from V2 authenticity training.
# ============================================================

AUTHENTICITY_SUPPORTED_DENOMINATIONS = {
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
}


# ============================================================
# IMAGE TRANSFORM
#
# Must match MobileNetV2 V2 evaluation preprocessing.
# ============================================================

AUTHENTICITY_TRANSFORM = transforms.Compose([

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
# LOAD YOLO MODEL
# ============================================================

if not YOLO_MODEL_PATH.exists():

    raise FileNotFoundError(
        "YOLO11 model not found:\n"
        f"{YOLO_MODEL_PATH}"
    )


yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)


# ============================================================
# LOAD AUTHENTICITY MODEL
# ============================================================

if not AUTHENTICITY_MODEL_PATH.exists():

    raise FileNotFoundError(
        "Authenticity model not found:\n"
        f"{AUTHENTICITY_MODEL_PATH}"
    )


checkpoint = torch.load(
    AUTHENTICITY_MODEL_PATH,
    map_location=DEVICE,
)


# ------------------------------------------------------------
# Recreate the SAME MobileNetV2 architecture used during V2
# training.
# ------------------------------------------------------------

authenticity_model = models.mobilenet_v2(
    weights=None
)


# Fine-tuned V2 architecture has a 2-class classifier.

authenticity_model.classifier[1] = nn.Linear(
    authenticity_model.classifier[1].in_features,
    2,
)


authenticity_model.load_state_dict(
    checkpoint["model_state_dict"]
)


authenticity_model = (
    authenticity_model.to(DEVICE)
)


authenticity_model.eval()


# ============================================================
# HELPER
# ============================================================

def _clamp_box(
    box,
    width,
    height,
):
    """
    Keep YOLO bounding box inside the image.
    """

    x1, y1, x2, y2 = box


    x1 = max(
        0,
        min(
            int(round(x1)),
            width - 1,
        ),
    )


    y1 = max(
        0,
        min(
            int(round(y1)),
            height - 1,
        ),
    )


    x2 = max(
        1,
        min(
            int(round(x2)),
            width,
        ),
    )


    y2 = max(
        1,
        min(
            int(round(y2)),
            height,
        ),
    )


    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def detect_currency(
    image_path,
    yolo_conf=0.25,
    yolo_iou=0.45,
):
    """
    Run complete currency detection.

    Parameters
    ----------
    image_path:
        Path to uploaded image.

    yolo_conf:
        Minimum YOLO detection confidence.

    yolo_iou:
        YOLO NMS IoU threshold.

    Returns
    -------
    dict
        Detection results for the frontend.
    """


    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")


    image_width, image_height = (
        image.size
    )


    # ========================================================
    # YOLO11 DETECTION
    # ========================================================

    yolo_results = yolo_model.predict(
        source=str(image_path),
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=640,
        device="cpu",
        verbose=False,
    )


    result = yolo_results[0]


    # ========================================================
    # NO NOTE DETECTED
    # ========================================================

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return {
            "success": False,
            "message": (
                "No Indian currency note "
                "was detected in the image."
            ),
            "denomination": None,
            "denomination_confidence": 0.0,
            "authenticity": None,
            "authenticity_confidence": 0.0,
            "box": None,
            "crop": None,
        }


    # ========================================================
    # SELECT HIGHEST-CONFIDENCE NOTE
    # ========================================================

    best_index = int(
        result.boxes.conf.argmax().item()
    )


    yolo_class_id = int(
        result.boxes.cls[
            best_index
        ].item()
    )


    yolo_confidence = float(
        result.boxes.conf[
            best_index
        ].item()
    )


    # Safety check

    if (
        yolo_class_id < 0
        or
        yolo_class_id >= len(
            YOLO_CLASS_NAMES
        )
    ):

        return {
            "success": False,
            "message": (
                "YOLO detected an unknown "
                "currency class."
            ),
            "denomination": None,
            "denomination_confidence": 0.0,
            "authenticity": None,
            "authenticity_confidence": 0.0,
            "box": None,
            "crop": None,
        }


    denomination = (
        YOLO_CLASS_NAMES[
            yolo_class_id
        ]
    )


    # ========================================================
    # GET BOUNDING BOX
    # ========================================================

    raw_box = (
        result.boxes.xyxy[
            best_index
        ].cpu().numpy()
    )


    x1, y1, x2, y2 = _clamp_box(
        raw_box,
        image_width,
        image_height,
    )


    # ========================================================
    # INVALID BOX
    # ========================================================

    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return {
            "success": False,
            "message": (
                "Currency note was detected "
                "but the bounding box was invalid."
            ),
            "denomination": denomination,
            "denomination_confidence": (
                yolo_confidence
            ),
            "authenticity": None,
            "authenticity_confidence": 0.0,
            "box": None,
            "crop": None,
        }


    # ========================================================
    # CROP NOTE
    # ========================================================

    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )


    # ========================================================
    # AUTHENTICITY CHECK
    # ========================================================
    #
    # ₹2000 was NOT included in V2 training.
    #
    # Therefore we must NOT pretend to classify ₹2000
    # as REAL/FAKE using this model.
    # ========================================================

    if (
        denomination
        not in AUTHENTICITY_SUPPORTED_DENOMINATIONS
    ):

        return {
            "success": True,
            "message": (
                "₹2000 detected. "
                "Authenticity classification "
                "is not available because "
                "₹2000 was excluded from "
                "the authenticity model."
            ),
            "denomination": denomination,
            "denomination_confidence": (
                yolo_confidence
            ),
            "authenticity": "UNSUPPORTED",
            "authenticity_confidence": 0.0,
            "box": (
                x1,
                y1,
                x2,
                y2,
            ),
            "crop": crop,
        }


    # ========================================================
    # PREPARE CROP
    # ========================================================

    input_tensor = (
        AUTHENTICITY_TRANSFORM(
            crop
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    # ========================================================
    # MOBILENETV2
    # ========================================================

    with torch.no_grad():

        outputs = authenticity_model(
            input_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1,
        )


        predicted_id = int(
            probabilities.argmax(
                dim=1
            ).item()
        )


        authenticity_confidence = float(
            probabilities[
                0,
                predicted_id
            ].item()
        )


    authenticity = (
        AUTHENTICITY_CLASS_NAMES[
            predicted_id
        ]
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "success": True,

        "message": (
            "Currency note detected "
            "and authenticity checked."
        ),

        "denomination": denomination,

        "denomination_confidence": (
            yolo_confidence
        ),

        "authenticity": authenticity,

        "authenticity_confidence": (
            authenticity_confidence
        ),

        "box": (
            x1,
            y1,
            x2,
            y2,
        ),

        "crop": crop,
    }


# ============================================================
# SIMPLE COMMAND-LINE TEST
#
# Run:
#
# python .\scripts\app_inference.py
#
# It will test the first image inside:
#
# test_images/
# ============================================================

if __name__ == "__main__":

    TEST_IMAGES_DIR = (
        PROJECT_ROOT
        / "test_images"
    )


    if not TEST_IMAGES_DIR.exists():

        print(
            "test_images folder not found:"
        )

        print(
            TEST_IMAGES_DIR
        )

        raise SystemExit(1)


    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


    image_paths = sorted(
        [
            p
            for p in TEST_IMAGES_DIR.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower()
                in image_extensions
            )
        ]
    )


    if not image_paths:

        print(
            "No images found inside:"
        )

        print(
            TEST_IMAGES_DIR
        )

        raise SystemExit(1)


    test_image = image_paths[0]


    print()
    print("=" * 70)
    print("INDIAN CURRENCY INFERENCE TEST")
    print("=" * 70)

    print()

    print(
        f"Image: {test_image.name}"
    )

    print()


    result = detect_currency(
        test_image
    )


    print(
        f"Success      : "
        f"{result['success']}"
    )


    print(
        f"Denomination : "
        f"₹{result['denomination']}"
        if result["denomination"]
        else
        "Denomination : Not detected"
    )


    if result["denomination"]:

        print(
            f"Denom. Conf. : "
            f"{result['denomination_confidence'] * 100:.2f}%"
        )


    print(
        f"Authenticity : "
        f"{result['authenticity']}"
    )


    if result["authenticity"]:

        print(
            f"Auth. Conf.  : "
            f"{result['authenticity_confidence'] * 100:.2f}%"
        )


    print()

    print(
        result["message"]
    )

    print()

    print("=" * 70)
