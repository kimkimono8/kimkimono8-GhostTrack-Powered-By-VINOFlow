from ultralytics import YOLO
from pathlib import Path

# -------------------------------
# ⚙️ CONFIG
# -------------------------------
MODEL_NAME = "yolov8n.pt"  # ✅ ใช้ base model แบบ clean ที่ไม่มีคลาสฝัง
DATA_YAML = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/dataset/dataset.yaml"

EPOCHS = 300
IMG_SIZE = 1280
BATCH_SIZE = 8
DEVICE = 0  # หรือ 'cuda:0'

PROJECT_NAME = "runs"
RUN_NAME = "train_GhostTrack_Clean"


# -------------------------------
# 🚀 TRAINING
# -------------------------------
def train():
    print("🚀 เริ่มการเทรน YOLOv8n แบบคลีนจาก 0 ...")

    if not Path(DATA_YAML).exists():
        print(f"[❌] ไม่พบ data.yaml: {DATA_YAML}")
        return

    # ✅ โหลด base YOLOv8 model แบบ clean โดยไม่ preload .pt ที่เคยฝัง class เดิม
    model = YOLO(MODEL_NAME)

    try:
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            pretrained=False,  # ✅ สำคัญ: เทรนจาก 0 ไม่ใช้ class structure เดิม
            lr0=0.002,
            lrf=0.0001,
            warmup_epochs=10,
            warmup_bias_lr=0.05,
            weight_decay=0.0008,
            cos_lr=True,
            momentum=0.97,
            dropout=0.2,
            patience=0,
            multi_scale=True,
            rect=True,
            hsv_h=0.02,
            hsv_s=0.6,
            hsv_v=0.3,
            scale=0.4,
            fliplr=0.5,
            flipud=0.1,
            mosaic=0.2,
            mixup=0.2,
            cutmix=0.1,
            cache="disk",
            plots=True,
            save=True,
            save_period=1,
            project=PROJECT_NAME,
            name=RUN_NAME,
            resume=False,
            verbose=True,
        )

    except Exception as e:
        print(f"[🔥 ERROR] Training failed: {e}")
        return

    print(f"✅ เทรนเสร็จแล้ว! ผลลัพธ์อยู่ที่: {PROJECT_NAME}/{RUN_NAME}/weights")


if __name__ == "__main__":
    train()
