from ultralytics import YOLO
import os

# -------------------------------
# ⚙️ CONFIGURATION
# -------------------------------
MODEL_NAME = 'yolov8n.pt'           # เลือก YOLOv8 Nano
DATA_YAML = 'custom_data.yaml'     # เส้นทางไฟล์ YAML
EPOCHS = 100                        # จำนวนรอบเทรน
IMG_SIZE = 640                     # ขนาดภาพ (ปรับได้)
BATCH_SIZE = 8                     # ขึ้นกับ GPU
DEVICE = 0                         # 0=GPU, 'cpu'=CPU

from ultralytics import YOLO
from pathlib import Path

# -------------------------------
# ⚙️ CONFIG
# -------------------------------
MODEL_NAME = "yolov8n.pt"
DATA_YAML = "data.yaml"
EPOCHS = 50
IMG_SIZE = 416
BATCH_SIZE = 8
DEVICE = 0  # หรือ "cuda:0"

# -------------------------------
# 🚀 TRAINING
# -------------------------------
def train():
    print("🚀 เริ่มการเทรน YOLOv8n ...")

    if not Path(MODEL_NAME).exists():
        print(f"[❌] ไม่พบไฟล์ model: {MODEL_NAME}")
        return

    if not Path(DATA_YAML).exists():
        print(f"[❌] ไม่พบ data.yaml: {DATA_YAML}")
        return

    model = YOLO(MODEL_NAME)

    try:
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project='runs',
            name='train_yolov8n_demo',
            resume=True,  # 🟢 เพิ่มตรงนี้
            verbose=True
        )

    except Exception as e:
        print(f"[🔥 ERROR] Training failed: {e}")
        return

    print("✅ เทรนเสร็จแล้ว! ไฟล์จะอยู่ใน runs/train_yolov8n_demo/weights")

if __name__ == '__main__':
    train()
