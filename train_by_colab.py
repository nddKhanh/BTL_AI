import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

class LeafCNN(nn.Module):
    def __init__(self, num_classes):
        super(LeafCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


base_dir = "/content/drive/MyDrive/BTL_AI/dataset_split_2"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=val_transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

num_classes = len(train_dataset.classes)
model = LeafCNN(num_classes=num_classes).to(device)

print("Classes:", train_dataset.classes)
print("Num classes:", num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

train_losses_list = []
val_losses_list = []
train_acc_list = []
val_acc_list = []


# Train
num_epochs = 20
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\n Epoch {epoch+1}/{num_epochs}")

    # Train
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training")

    for images, labels in train_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_bar.set_postfix(loss=loss.item(), acc=correct / total)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total


    # Valid
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(valid_loader)
    val_acc = val_correct / val_total


    # Save model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "/content/drive/MyDrive/BTL_AI/best_model.pth")
        print("Best Acc:", best_acc)

    scheduler.step()


    print(f"""
Epoch [{epoch+1}/{num_epochs}]
Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}
Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}
Best Acc:   {best_acc:.4f}
LR:        {scheduler.get_last_lr()[0]:.6f}
""")


    # Lưu trữ Metrics
    train_losses_list.append(train_loss)
    val_losses_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

metrics = {
    "train_losses": train_losses_list,
    "val_losses": val_losses_list,
    "train_acc": train_acc_list,
    "val_acc": val_acc_list,
    "best_acc": best_acc
}

np.savez(
    os.path.join("/content/drive/MyDrive/BTL_AI/metrics", "metrics.npz"),
    train_losses=np.array(train_losses_list),
    val_losses=np.array(val_losses_list),
    train_acc=np.array(train_acc_list),
    val_acc=np.array(val_acc_list)
)


# confusion matrix

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

# Save
np.savez(
    os.path.join(
        "/content/drive/MyDrive/BTL_AI/metrics",
        "confusion_matrix_data.npz"
    ),
    y_true=np.array(all_labels),
    y_pred=np.array(all_preds),
    class_names=np.array(train_dataset.classes)
)