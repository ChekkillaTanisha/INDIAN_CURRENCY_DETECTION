from pathlib import Path
import shutil

GT_DIR = Path(r"dataset\yolo11\labels\val")
PRED_DIR = Path(r"C:\Users\Tanisha\runs\detect\predict\labels")
IMG_DIR = Path(r"dataset\yolo11\images\val")
OUT_DIR = Path(r"error_analysis")

MISSED_100 = OUT_DIR / "missed_100"
FALSE_POS = OUT_DIR / "false_positives"

MISSED_100.mkdir(parents=True, exist_ok=True)
FALSE_POS.mkdir(parents=True, exist_ok=True)

def read_label(path):
    if not path.exists():
        return []
    rows = []
    text = path.read_text().strip()
    if not text:
        return rows

    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5:
            cls = int(float(p[0]))
            x, y, w, h = map(float, p[1:5])
            conf = float(p[5]) if len(p) >= 6 else None
            rows.append((cls, x, y, w, h, conf))
    return rows

def xyxy(x, y, w, h):
    return (x-w/2, y-h/2, x+w/2, y+h/2)

def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a[1], a[2], a[3], a[4])
    bx1, by1, bx2, by2 = xyxy(b[1], b[2], b[3], b[4])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2-ix1)
    ih = max(0, iy2-iy1)
    inter = iw * ih

    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0

# Class mapping from your dataset:
# 0=10, 1=20, 2=50, 3=100, 4=200, 5=500, 6=2000
names = {
    0: "10",
    1: "20",
    2: "50",
    3: "100",
    4: "200",
    5: "500",
    6: "2000"
}

missed_100 = []
false_positive_files = []

for gt_file in GT_DIR.glob("*.txt"):
    pred_file = PRED_DIR / gt_file.name

    gt = read_label(gt_file)
    pred = read_label(pred_file)

    # Find every GT object and whether it has a matching prediction
    matched_pred = set()

    for gi, g in enumerate(gt):
        best_iou = 0
        best_pi = None

        for pi, p in enumerate(pred):
            if pi in matched_pred:
                continue

            if p[0] != g[0]:
                continue

            score = iou(g, p)

            if score > best_iou:
                best_iou = score
                best_pi = pi

        if best_iou >= 0.50 and best_pi is not None:
            matched_pred.add(best_pi)
        else:
            # Class 3 = ₹100
            if g[0] == 3:
                missed_100.append((gt_file.name, best_iou))

    # Any prediction that did not match a GT object = false positive
    unmatched = [p for pi, p in enumerate(pred) if pi not in matched_pred]

    if unmatched:
        false_positive_files.append((gt_file.name, len(unmatched)))

# Copy missed ₹100 images
for filename, score in missed_100:
    stem = Path(filename).stem

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img = IMG_DIR / (stem + ext)
        if img.exists():
            shutil.copy2(img, MISSED_100 / img.name)
            break

# Copy false-positive images
for filename, count in false_positive_files:
    stem = Path(filename).stem

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img = IMG_DIR / (stem + ext)
        if img.exists():
            shutil.copy2(img, FALSE_POS / img.name)
            break

print()
print("=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)
print()
print("Missed ₹100 images:", len(missed_100))
print("False-positive images:", len(false_positive_files))
print()
print("Missed ₹100 images:")
for filename, score in missed_100:
    print(f"  {filename}   best IoU={score:.3f}")
print()
print("False-positive images:")
for filename, count in false_positive_files:
    print(f"  {filename}   unmatched predictions={count}")
print()
print("Output folder:")
print(OUT_DIR.resolve())
print()
print("Missed ₹100 folder:")
print(MISSED_100.resolve())
print()
print("False-positive folder:")
print(FALSE_POS.resolve())
print("=" * 70)
