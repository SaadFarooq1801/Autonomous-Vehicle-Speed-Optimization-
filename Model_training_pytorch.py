# PyTorch version — supports native Windows GPU (CUDA).
# Install PyTorch with CUDA before running:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_SIZE = 96
BATCH_SIZE = 32
LR_HEAD = 0.001     # phase 1 
LR_FINETUNE = 1e-5  # phase 2 
EPOCHS_HEAD = 10    # phase 1 
EPOCHS_FINETUNE = 15  # phase 2 
PATIENCE = 5

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_data(data_path):
    images, labels = [], []
    classes = sorted(c for c in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, c)))
    print(f"Found {len(classes)} classes: {classes}")

    for class_num, class_folder in enumerate(classes):
        folder_path = os.path.join(data_path, class_folder)
        count = 0
        for image_file in os.listdir(folder_path):
            if image_file.startswith('._') or not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            try:
                img = cv2.imread(os.path.join(folder_path, image_file))
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(class_num)
                count += 1
            except Exception as e:
                print(f"Error loading {image_file}: {e}")
        print(f"  {class_folder}: {count} images")

    return np.array(images), np.array(labels), classes


class SpeedSignDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze entire backbone — with ~500 images/class, fine-tuning conv layers causes overfitting.
    # Only the classifier head is trained from scratch.
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def run_phase(model, train_loader, test_loader, criterion, optimizer, epochs, phase_label, history, best_val_acc, patience_counter):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    print(f"\n── {phase_label} ──")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{epochs}  loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> Saved best model (val_acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return best_val_acc, patience_counter


if __name__ == "__main__":
    # Merge Train + Test then re-split 80/20 — original split was inverted (3k train vs 10k test)
    X_all, y_all, class_names = load_data(os.path.join(DATA_DIR, "Train"))
    X_extra, y_extra, _ = load_data(os.path.join(DATA_DIR, "Test"))
    X_all = np.concatenate([X_all, X_extra])
    y_all = np.concatenate([y_all, y_extra])
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    num_classes = len(class_names)
    train_loader = DataLoader(SpeedSignDataset(X_train, y_train, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(SpeedSignDataset(X_test, y_test, test_transform),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    # Phase 1 — head only, backbone fully frozen
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR_HEAD, weight_decay=1e-4)
    best_val_acc, patience_counter = run_phase(
        model, train_loader, test_loader, criterion, optimizer,
        EPOCHS_HEAD, "Phase 1: training head only", history, 0.0, 0
    )

    phase2_start = len(history["train_acc"])

    # Phase 2 — unfreeze last 2 backbone blocks, fine-tune with very low LR
    for block in list(model.features.children())[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR_FINETUNE, weight_decay=1e-4)
    best_val_acc, _ = run_phase(
        model, train_loader, test_loader, criterion, optimizer,
        EPOCHS_FINETUNE, "Phase 2: fine-tuning last 2 backbone blocks", history, best_val_acc, 0
    )

    model.load_state_dict(torch.load("best_model.pt"))
    torch.save(model, "model_trained.pt")
    print("Model saved successfully!")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.axvline(x=phase2_start - 0.5, color='gray', linestyle='--', label='Phase 2 start')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.axvline(x=phase2_start - 0.5, color='gray', linestyle='--', label='Phase 2 start')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    test_dataset = SpeedSignDataset(X_test, y_test, test_transform)
    sample_images = torch.stack([test_dataset[i][0] for i in range(5)]).to(device)
    sample_labels = [test_dataset[i][1] for i in range(5)]
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(sample_images), dim=1)
    for i in range(5):
        pred = probs[i].argmax().item()
        conf = probs[i][pred].item() * 100
        print(f"Image {i+1}: Predicted={class_names[pred]}, Actual={class_names[sample_labels[i]]}, Confidence={conf:.2f}%")
