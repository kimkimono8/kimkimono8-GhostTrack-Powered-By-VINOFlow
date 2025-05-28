from ultralytics import YOLO

model = YOLO("D:/yolov8n_train/yolov8n.pt")
print("✅ Number of classes:", model.model.model[-1].nc)