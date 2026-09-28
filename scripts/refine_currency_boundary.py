from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# YOLO11 + OpenCV Currency Boundary Refinement TEST
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "currency_yolo11n-3"
    / "weights"
    / "best.pt"
)

INPUT_DIR = PROJECT_ROOT / "dataset" / "yolo11" / "images" / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "boundary_test"
REFINED_DIR = OUTPUT_DIR / "refined"
VIS_DIR = OUTPUT_DIR / "visual_check"

REFINED_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Find note contour inside YOLO detection
# ============================================================

def find_note_contour(roi):

    if roi is None or roi.size == 0:
        return None

    h, w = roi.shape[:2]

    # Slight blur reduces tiny printed/text edges.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 40, 120)

    # Close small gaps in the note border.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Find external contours.
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    roi_area = float(w * h)

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < roi_area * 0.15:
            continue

        if area > roi_area * 1.05:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        # Prefer quadrilateral contours.
        if len(approx) == 4:

            approx_area = cv2.contourArea(approx)

            if approx_area >= roi_area * 0.25:
                candidates.append(
                    (approx_area, approx)
                )

    if candidates:

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return candidates[0][1]

    # Fallback: largest reasonable contour.
    reasonable = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if roi_area * 0.25 <= area <= roi_area * 1.05:
            reasonable.append((area, contour))

    if reasonable:

        reasonable.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return reasonable[0][1]

    return None


# ============================================================
# Perspective correction
# ============================================================

def order_points(points):

    points = np.array(points, dtype=np.float32)

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]

    top_right = points[np.argmin(d)]
    bottom_left = points[np.argmax(d)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ],
        dtype=np.float32
    )


def perspective_crop(roi, contour):

    if contour is None:
        return None

    if len(contour) != 4:
        return None

    points = contour.reshape(4, 2)
    rect = order_points(points)

    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    if width < 50 or height < 30:
        return None

    # Keep reasonable dimensions.
    if width > 4000 or height > 4000:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        roi,
        matrix,
        (width, height)
    )

    return warped


# ============================================================
# YOLO11
# ============================================================

print("=" * 70)
print("YOLO11 + CURRENCY BOUNDARY REFINEMENT TEST")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{INPUT_DIR}"
    )

print()
print(f"Model : {MODEL_PATH}")
print(f"Input : {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

model = YOLO(str(MODEL_PATH))

image_paths = sorted(
    list(INPUT_DIR.glob("*.jpg"))
    + list(INPUT_DIR.glob("*.jpeg"))
    + list(INPUT_DIR.glob("*.png"))
)

print(f"Images found: {len(image_paths)}")
print()

successful = 0
fallback = 0
failed = 0

# ============================================================
# Process images
# ============================================================

for index, image_path in enumerate(image_paths, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        failed += 1
        continue

    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        failed += 1
        continue

    # Select highest confidence detection.
    confidence = result.boxes.conf.cpu().numpy()
    best_index = int(np.argmax(confidence))

    box = (
        result.boxes.xyxy[best_index]
        .cpu()
        .numpy()
        .astype(int)
    )

    x1, y1, x2, y2 = box

    h, w = image.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        failed += 1
        continue

    roi = image[y1:y2, x1:x2].copy()

    contour = find_note_contour(roi)

    refined = perspective_crop(
        roi,
        contour
    )

    # ========================================================
    # Visual diagnostic
    # ========================================================

    visual = image.copy()

    # Original YOLO box = BLUE
    cv2.rectangle(
        visual,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        4
    )

    label = (
        f"YOLO "
        f"{float(confidence[best_index]):.2f}"
    )

    cv2.putText(
        visual,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # ========================================================
    # Draw refined contour
    # ========================================================

    if contour is not None:

        contour_global = contour.copy()

        contour_global[:, 0, 0] += x1
        contour_global[:, 0, 1] += y1

        # Refined boundary = GREEN
        cv2.polylines(
            visual,
            [contour_global],
            True,
            (0, 255, 0),
            5
        )

    # Save visual diagnostic
    visual_path = VIS_DIR / image_path.name

    cv2.imwrite(
        str(visual_path),
        visual
    )

    # ========================================================
    # Save refined crop
    # ========================================================

    if refined is not None:

        refined_path = REFINED_DIR / image_path.name

        cv2.imwrite(
            str(refined_path),
            refined
        )

        successful += 1

    else:

        # Fallback: save YOLO ROI.
        fallback_path = REFINED_DIR / image_path.name

        cv2.imwrite(
            str(fallback_path),
            roi
        )

        fallback += 1

    if index % 50 == 0 or index == len(image_paths):

        print(
            f"Processed {index}/{len(image_paths)} | "
            f"refined={successful} | "
            f"fallback={fallback} | "
            f"failed={failed}"
        )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("BOUNDARY TEST COMPLETE")
print("=" * 70)
print()
print(f"Total images       : {len(image_paths)}")
print(f"Refined boundaries : {successful}")
print(f"Fallback YOLO crop : {fallback}")
print(f"Failed             : {failed}")
print()
print(f"Visual results:")
print(VIS_DIR)
print()
print(f"Refined crops:")
print(REFINED_DIR)
print()
print("IMPORTANT:")
print("BLUE = original YOLO11 box")
print("GREEN = proposed refined currency boundary")
print()
print("Do NOT train again yet.")
print("Inspect the GREEN boundaries first.")