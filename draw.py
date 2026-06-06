import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load data
data = np.load("metrics.npz")
metrics_data = np.load("confusion_matrix_data.npz")

# DataFrame metrics
df = pd.DataFrame({
    "train_loss": data["train_losses"],
    "val_loss": data["val_losses"],
    "train_acc": data["train_acc"],
    "val_acc": data["val_acc"],
})

# Plot metrics
df[["train_loss", "val_loss", "train_acc", "val_acc"]].plot()

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Metrics")
plt.grid(True)
plt.show()

print(df.head())


# ====== CONFUSION MATRIX ======
# # giả sử file lưu key là "cm"
# print(metrics_data.files)
# print(metrics_data["y_true"])
# print(metrics_data["y_pred"])
# print(metrics_data["class_names"])

y_true = metrics_data["y_true"]
y_pred = metrics_data["y_pred"]
class_names = metrics_data["class_names"]

cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(7,6))
plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.colorbar()

# tick labels (tên class)
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color=color, fontsize=12)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()

# plt.figure(figsize=(6, 5))
# plt.imshow(cm, cmap="Blues")
# plt.title("Confusion Matrix")
# plt.colorbar()
#
# # hiển thị số trong từng ô
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, cm[i, j],
#                  ha="center", va="center",
#                  color="black")
#
# plt.xlabel("Predicted label")
# plt.ylabel("True label")
# plt.show()