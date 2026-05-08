import cv2
import torch
import numpy as np
from torchvision import transforms
import os
import time
from collections import deque

MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_trained.pt")
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "Train")
IMG_SIZE    = 96
CONF_THRESH = 0.60
SMOOTH_N    = 8     # frames to average predictions over

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = sorted(c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c)))

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
print(f"Model on {device} | Classes: {class_names}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_roi(bgr_crop):
    """Match dataset conditions: CLAHE normalisation then standard ImageNet transforms."""
    lab   = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    bgr   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def predict(rgb_crop):
    tensor = transform(rgb_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probs


def draw_ui(frame, probs, preview_rgb):
    h, w = frame.shape[:2]

    # ── ROI box ───────────────────────────────────────────────────────────────
    roi_size = min(h, w) // 2
    cx, cy   = w // 2, h // 2
    x1 = cx - roi_size // 2
    y1 = cy - roi_size // 2
    x2 = cx + roi_size // 2
    y2 = cy + roi_size // 2

    best_idx  = int(np.argmax(probs))
    best_conf = float(probs[best_idx])
    confident = best_conf >= CONF_THRESH
    roi_color = (0, 220, 0) if confident else (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
    cv2.putText(frame, "Fill box with sign", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 1)

    # ── Main prediction label ─────────────────────────────────────────────────
    label = f"{class_names[best_idx]} km/h" if confident else "???"
    cv2.putText(frame, label, (18, 65),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, roi_color, 3)
    cv2.putText(frame, f"{best_conf * 100:.1f}% confidence", (18, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, roi_color, 2)
    if not confident:
        cv2.putText(frame, "LOW CONFIDENCE", (18, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── Confidence bars ───────────────────────────────────────────────────────
    px, py    = w - 230, 20
    bar_max_w = 160
    bar_h     = 16
    stride    = 36

    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 12, py - 18),
                  (w - 5, py + len(class_names) * stride + 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "Confidence", (px - 8, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for i, (cls, prob) in enumerate(zip(class_names, probs)):
        by      = py + i * stride + 14
        filled  = int(prob * bar_max_w)
        is_best = (i == best_idx)

        cv2.rectangle(frame, (px, by), (px + bar_max_w, by + bar_h), (60, 60, 60), -1)
        bar_color = (0, 220, 0) if is_best and confident else \
                    (0, 165, 255) if is_best else (100, 180, 255)
        cv2.rectangle(frame, (px, by), (px + filled, by + bar_h), bar_color, -1)
        cv2.rectangle(frame, (px, by), (px + bar_max_w, by + bar_h), (120, 120, 120), 1)

        text  = f"{cls} km/h  {prob * 100:.1f}%"
        color = (255, 255, 255) if is_best else (180, 180, 180)
        cv2.putText(frame, text, (px, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # ── Model preview (what the model actually sees, bottom-left) ─────────────
    preview_bgr    = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    preview_disp   = cv2.resize(preview_bgr, (128, 128), interpolation=cv2.INTER_NEAREST)
    ph, pw         = preview_disp.shape[:2]
    margin         = 10
    py_pos         = h - ph - margin
    frame[py_pos:py_pos + ph, margin:margin + pw] = preview_disp
    cv2.rectangle(frame, (margin - 1, py_pos - 1),
                  (margin + pw, py_pos + ph), (200, 200, 200), 1)
    cv2.putText(frame, "Model input", (margin, py_pos - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prob_buffer = deque(maxlen=SMOOTH_N)
    prev_time   = time.time()
    print("Running — press Q to quit, S to save current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w    = frame.shape[:2]
        roi_sz  = min(h, w) // 2
        cx, cy  = w // 2, h // 2
        x1 = cx - roi_sz // 2
        y1 = cy - roi_sz // 2
        x2 = cx + roi_sz // 2
        y2 = cy + roi_sz // 2

        roi_crop    = frame[y1:y2, x1:x2]
        preview_rgb = preprocess_roi(roi_crop)   # CLAHE-normalised RGB
        probs       = predict(preview_rgb)

        prob_buffer.append(probs)
        smooth_probs = np.mean(prob_buffer, axis=0).copy()  # temporal average

        # Arabic ٣ (3) resembles ٨ (8) in some sign fonts — boost '30' so it
        # wins the 30/80 toss-up. Only affects that one pair; all other classes untouched.
        idx_30 = class_names.index('30')
        idx_80 = class_names.index('80')
        if smooth_probs[idx_80] > smooth_probs[idx_30]:
            smooth_probs[idx_30] *= 1.5
            smooth_probs /= smooth_probs.sum()   # renormalise

        draw_ui(frame, smooth_probs, preview_rgb)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (18, h - 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow("Speed Sign Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
