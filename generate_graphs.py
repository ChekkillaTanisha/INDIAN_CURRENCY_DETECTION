import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "runs/authenticity_new/training_history.csv"
)

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy")
plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_graph.png")
plt.close()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
plt.close()

print("Graphs Generated Successfully")
