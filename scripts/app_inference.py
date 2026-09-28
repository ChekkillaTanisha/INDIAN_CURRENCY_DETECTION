import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from blockchain import add_record
from ocr_serial import extract_serial_number


import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# INDIAN CURRENCY DETECTION SYSTEM
#
# YOLO11:
#   Detects denomination.
#
# MobileNetV2:
#   Detects REAL vs FAKE.
#
# FINAL SYSTEM:
#   YOLO11 + MobileNetV2
#
# YOLO11 answers:
#   "Which denomination is this?"
#
# MobileNetV2 answers:
#   "Is this note REAL or FAKE?"
#
# Both results are combined into the final application result.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "currency_yolo11n-3"
    / "weights"
    / "best.pt"
)


AUTHENTICITY_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "authenticity_new"
    / "best_authenticity_mobilenetv2_new.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# YOLO11 DENOMINATION CLASSES
#
# MUST MATCH YOLO11 TRAINING CLASS ORDER
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
# MobileNetV2 AUTHENTICITY CLASSES
#
# 0 = FAKE
# 1 = REAL
# ============================================================

AUTHENTICITY_CLASS_NAMES = [
    "FAKE",
    "REAL",
]


# ============================================================
# AUTHENTICITY THRESHOLD
#
# The NEW MobileNetV2 was trained on:
#
# ₹10
# ₹20
# ₹50
# ₹100
# ₹200
# ₹500
# ₹2000
#
# Therefore ALL denominations are supported.
#
# The new model was trained with class weights and the
# final decision uses P(FAKE) >= 0.50.
# ============================================================

AUTHENTICITY_THRESHOLD = 0.50


# ============================================================
# IMAGE TRANSFORM
# ============================================================

AUTHENTICITY_TRANSFORM = transforms.Compose(
    [
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
    ]
)


# ============================================================
# LOAD YOLO11
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
# LOAD NEW MobileNetV2
# ============================================================

if not AUTHENTICITY_MODEL_PATH.exists():

    raise FileNotFoundError(
        "New MobileNetV2 authenticity model not found:\n"
        f"{AUTHENTICITY_MODEL_PATH}"
    )


checkpoint = torch.load(
    AUTHENTICITY_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

print(type(checkpoint))
print(checkpoint.keys())


# ============================================================
# RECREATE MobileNetV2
# ============================================================

authenticity_model = models.mobilenet_v2(
    weights=None
)


authenticity_model.classifier[1] = nn.Linear(
    authenticity_model.classifier[1].in_features,
    2,
)


authenticity_model.load_state_dict(
    checkpoint["model_state_dict"]
)


authenticity_model = authenticity_model.to(
    DEVICE
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
# MobileNetV2 AUTHENTICITY CLASSIFICATION
# ============================================================

def _classify_authenticity(
    crop,
):

    input_tensor = (
        AUTHENTICITY_TRANSFORM(
            crop
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    with torch.no_grad():

        outputs = authenticity_model(
            input_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1,
        )


    fake_probability = float(
        probabilities[
            0,
            0,
        ].item()
    )


    real_probability = float(
        probabilities[
            0,
            1,
        ].item()
    )


    # --------------------------------------------------------
    # FINAL AUTHENTICITY DECISION
    # --------------------------------------------------------

    if (
        fake_probability
        >= AUTHENTICITY_THRESHOLD
    ):

        authenticity = "FAKE"

        authenticity_confidence = (
            fake_probability
        )

    else:

        authenticity = "REAL"

        authenticity_confidence = (
            real_probability
        )


    return (
        authenticity,
        authenticity_confidence,
        fake_probability,
        real_probability,
    )


# ============================================================
# MAIN INFERENCE
# ============================================================

def detect_currency(
    image_path,
    yolo_conf=0.40,
    yolo_iou=0.45,
):

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
    # STEP 1 — YOLO11 DENOMINATION DETECTION
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
                "was detected."
            ),

            "denomination": None,

            "denomination_confidence": 0.0,

            "authenticity": None,

            "authenticity_confidence": 0.0,

            "fake_probability": 0.0,

            "real_probability": 0.0,

            "combined_confidence": 0.0,

            "box": None,

            "crop": None,
        }


    # ========================================================
    # SELECT BEST YOLO DETECTION
    # ========================================================

    best_index = int(
    torch.argmax(
        result.boxes.conf
    ).item()
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


    # ========================================================
    # YOLO CLASS SAFETY CHECK
    # ========================================================

    # Reject ₹2000 notes completely
    if yolo_class_id == 6:

      return {
        "success": False,

        "message": "₹2000 notes are not supported.",
        
        "denomination": "2000",

        "denomination_confidence": yolo_confidence,

        "authenticity": None,

        "authenticity_confidence": 0.0,

        "fake_probability": 0.0,

        "real_probability": 0.0,

        "combined_confidence": 0.0,

        "box": None,

        "crop": None,
    }


    if (
    yolo_class_id < 0
    or
    yolo_class_id >= len(result.names)
):

      return {
        "success": False,

        "message": (
            "YOLO11 detected an unknown "
            "currency class."
        ),

        "denomination": None,

        "denomination_confidence": 0.0,

        "authenticity": None,

        "authenticity_confidence": 0.0,

        "fake_probability": 0.0,

        "real_probability": 0.0,

        "combined_confidence": 0.0,

        "box": None,

        "crop": None,
    }


    denomination = result.names[yolo_class_id]
    
    


    # ========================================================
    # GET YOLO BOUNDING BOX
    # ========================================================

    raw_box = (
        result.boxes.xyxy[
            best_index
        ]
        .cpu()
        .numpy()
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

            "fake_probability": 0.0,

            "real_probability": 0.0,

            "combined_confidence": 0.0,

            "box": None,

            "crop": None,
        }


    # ========================================================
    # STEP 2 — CROP NOTE FROM YOLO BOX
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
    # STEP 3 — MobileNetV2 AUTHENTICITY
    #
    # YOLO11 detected the denomination.
    #
    # MobileNetV2 now receives the YOLO-detected note crop
    # and determines REAL vs FAKE.
    # ========================================================

    (
        authenticity,
        authenticity_confidence,
        fake_probability,
        real_probability,
    ) = _classify_authenticity(
        crop
    )


    # ========================================================
    # STEP 4 — COMBINE YOLO11 + MobileNetV2
    #
    # This is NOT averaging the probabilities.
    #
    # YOLO confidence describes denomination confidence.
    # MobileNet confidence describes authenticity confidence.
    #
    # The conservative combined confidence is the lower of
    # the two, because BOTH decisions are required.
    # ========================================================

    combined_confidence = min(
        yolo_confidence,
        authenticity_confidence,
    )
    
    # ========================================================
    # OCR SERIAL NUMBER
    # ========================================================

    try:

        serial_number = extract_serial_number(
            str(image_path)
        )

    except Exception:

        serial_number = "Not Detected"


    # ========================================================
    # FINAL SYSTEM MESSAGE
    # ========================================================

    if authenticity == "REAL":

        final_message = (
            f"₹{denomination} detected by YOLO11 "
            f"and classified as REAL by MobileNetV2."
        )

    else:

        final_message = (
            f"₹{denomination} detected by YOLO11 "
            f"and classified as FAKE by MobileNetV2."
        )


    # ========================================================
    # FINAL COMBINED RESULT
    # ========================================================

    return {

        "success": True,

        "message": final_message,
    

        # ----------------------------------------------------
        # YOLO11 RESULT
        # ----------------------------------------------------

        "denomination": denomination,
        
        "serial_number": serial_number,

        "denomination_confidence": (
            yolo_confidence
        ),

        # ----------------------------------------------------
        # MobileNetV2 RESULT
        # ----------------------------------------------------

        "authenticity": authenticity,

        "authenticity_confidence": (
            authenticity_confidence
        ),

        "fake_probability": (
            fake_probability
        ),

        "real_probability": (
            real_probability
        ),

        # ----------------------------------------------------
        # COMBINED RESULT
        # ----------------------------------------------------

        "combined_confidence": (
            combined_confidence
        ),

        "model_combination": (
            "YOLO11 + MobileNetV2"
        ),

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        "authenticity_threshold": (
            AUTHENTICITY_THRESHOLD
        ),

        # ----------------------------------------------------
        # IMAGE INFORMATION
        # ----------------------------------------------------

        "box": (
            x1,
            y1,
            x2,
            y2,
        ),

        "crop": crop,
    }