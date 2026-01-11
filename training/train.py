import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --------------------
# Config
# --------------------
DATA_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# --------------------
# Transforms
# --------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

# --------------------
# Datasets
# --------------------
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train images:", len(train_ds))
print("Val images:", len(val_ds))
print("Classes:", train_ds.classes)

# --------------------
# Model
# --------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze backbone
# Unfreeze last N layers for fine-tuning
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last 4 blocks
for param in model.features[-4:].parameters():
    param.requires_grad = True


# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

model = model.to(DEVICE)

# --------------------
# Loss & Optimizer
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)


# --------------------
# Training Loop
# --------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---- Train ----
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ---- Validate ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Val Acc:   {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/food_model.pth")
        print("✅ Saved new best model")

print("\nTraining complete.")
print("Best Val Accuracy:", best_val_acc)
