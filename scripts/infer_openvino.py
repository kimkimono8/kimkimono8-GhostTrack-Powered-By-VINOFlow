import os
import cv2
import numpy as np
import time
from openvino.runtime import Core

# === CONFIG ===
MODEL_XML = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/models/yolov8n_ghosttrack_demo.xml"  # ใช้ static shape สำหรับความเร็ว
IMAGE_PATH = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/test.jpg"  # รูปภาพทดสอบ
DEVICE = "GPU"  # หรือ "GPU" ถ้ารองรับ

# === กำหนดค่า Environment Variables สำหรับการใช้งาน CPU ===
os.environ["OMP_NUM_THREADS"] = "8"  # กำหนดจำนวน Thread ที่จะใช้
os.environ["OPENVINO_CPU_THREADS_NUM"] = "8"  # CPU Thread จำนวนที่เหมาะสม
os.environ["OPENVINO_CPU_BIND_THREAD"] = "YES"  # Bind Thread กับ Core
os.environ["OPENVINO_CPU_THROUGHPUT_STREAMS"] = "1"  # ให้ใช้ Stream เดียว

# === LOAD MODEL ===
core = Core()
model = core.read_model(MODEL_XML)
compiled_model = core.compile_model(model, DEVICE)

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

_, c, h, w = input_layer.shape  # ex: (1, 3, 1280, 1280)

# === LOAD IMAGE ===
image = cv2.imread(IMAGE_PATH)
image_resized = cv2.resize(image, (w, h))
image_input = image_resized.transpose(2, 0, 1)[np.newaxis, ...]  # HWC → NCHW
image_input = image_input.astype(np.float32) / 255.0  # Normalize 0-1

# === INFERENCE ===
start = time.time()
result = compiled_model([image_input])[output_layer]
end = time.time()

# === PRINT RESULT ===
print(f"[INFO] Inference done in {end - start:.3f} seconds")
print(f"Output shape: {result.shape}")  # ex: (1, 11, 33600)

# === OPTIONAL: Postprocess (เช่น NMS, decode box, confidence filter) ===
# คุณสามารถเขียน postprocess ได้ตาม YOLOv8 หรือใช้ Ultralytics utils ช่วยก็ได้
