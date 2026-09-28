import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "runs/authenticity_new/test_predictions.csv"
)

tp = len(df[(df["true_label"]=="REAL") &
            (df["predicted_label"]=="REAL")])

tn = len(df[(df["true_label"]=="FAKE") &
            (df["predicted_label"]=="FAKE")])

fp = len(df[(df["true_label"]=="FAKE") &
            (df["predicted_label"]=="REAL")])

fn = len(df[(df["true_label"]=="REAL") &
            (df["predicted_label"]=="FAKE")])

cm = [
    [tp, fn],
    [fp, tn]
]

plt.figure(figsize=(6,5))
plt.imshow(cm)

plt.xticks([0,1], ["Real","Fake"])
plt.yticks([0,1], ["Real","Fake"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]),
                 ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

print("TP =", tp)
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)


accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp) if (tp + fp) else 0

recall = tp / (tp + fn) if (tp + fn) else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall)
    else 0
)

print("Accuracy =", accuracy)
print("Precision =", precision)
print("Recall =", recall)
print("F1 =", f1)