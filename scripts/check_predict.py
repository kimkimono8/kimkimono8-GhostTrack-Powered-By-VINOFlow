from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# โหลดโมเดล
model = YOLO(
    "D:/GhostTrack-Powered By VINOFlow/GhostTrack/best.pt"
)

# === PART 1: แสดงคลาส ===
print("==== CLASS LIST ====")
for i, name in model.names.items():
    print(f"{i}: {name}")

# === PART 2: รันภาพเดียว ===
image_path = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/debug.jpg"
results = model(image_path)
r = results[0]  # <<<<< ต้องมีบรรทัดนี้ก่อนใช้ r.plot()

print(f"\n==== INFERENCE RESULT ====")
print(f"Input shape: {r.orig_shape}")
print(f"Classes: {r.boxes.cls}")
print(f"Confidences: {r.boxes.conf}")
print(f"Bounding boxes: {r.boxes.xyxy}")

# === PART 3: แสดงผลภาพ ===
result_img = r.plot()
cv2.imwrite("result_debug.jpg", result_img)
print("Saved: result_debug.jpg")

# === PART 4: รันหลายภาพในโฟลเดอร์ ===
input_folder = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/scripts/test_images"
output_folder = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/scripts/results"
os.makedirs(output_folder, exist_ok=True)


for img_path in Path(input_folder).glob("*.jpg"):
    results = model(str(img_path))
    r = results[0]
    res_plot = r.plot()
    save_path = os.path.join(output_folder, img_path.name)  # ✅ แก้ตรงนี้
    cv2.imwrite(save_path, res_plot)
    print(f"Saved: {save_path}")
