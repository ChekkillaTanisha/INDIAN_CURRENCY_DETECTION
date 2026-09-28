import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "runs/currency_yolo11n-3/results.csv"
)

# Precision
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["metrics/precision(B)"])
plt.title("YOLOv11 Precision Curve")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("P_curve.png")
plt.close()

# Recall
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["metrics/recall(B)"])
plt.title("YOLOv11 Recall Curve")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.grid(True)
plt.savefig("R_curve.png")
plt.close()

# F1
precision = df["metrics/precision(B)"]
recall = df["metrics/recall(B)"]

f1 = 2 * precision * recall / (precision + recall)

plt.figure(figsize=(7,5))
plt.plot(df["epoch"], f1)
plt.title("YOLOv11 F1 Curve")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True)
plt.savefig("F1_curve.png")
plt.close()

# mAP50
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["metrics/mAP50(B)"])
plt.title("YOLOv11 mAP@50 Curve")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.grid(True)
plt.savefig("PR_curve.png")
plt.close()

print("All YOLO graphs generated successfully")