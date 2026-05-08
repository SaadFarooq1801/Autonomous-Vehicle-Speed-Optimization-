import torch
import numpy as np
import cv2
import os
from torchvision import transforms

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_trained.pt")
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "Test")
IMG_SIZE   = 96

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = sorted(c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c)))

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

correct = 0
total   = 0
per_class_correct = {c: 0 for c in class_names}
per_class_total   = {c: 0 for c in class_names}

for class_idx, cls in enumerate(class_names):
    folder = os.path.join(DATA_DIR, cls)
    files  = [f for f in os.listdir(folder)
              if not f.startswith('._') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for fname in files[:50]:   # test up to 50 images per class
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor  = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

        predicted = int(np.argmax(probs))
        is_correct = (predicted == class_idx)

        per_class_total[cls]   += 1
        per_class_correct[cls] += int(is_correct)
        correct += int(is_correct)
        total   += 1

print(f"\n{'Class':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("-" * 38)
for cls in class_names:
    t = per_class_total[cls]
    c = per_class_correct[cls]
    acc = (c / t * 100) if t else 0
    print(f"{cls:<8} {c:>8} {t:>8} {acc:>9.1f}%")

print("-" * 38)
print(f"{'TOTAL':<8} {correct:>8} {total:>8} {correct/total*100:>9.1f}%")
