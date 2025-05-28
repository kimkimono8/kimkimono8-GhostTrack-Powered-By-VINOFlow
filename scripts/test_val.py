from ultralytics import YOLO

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = YOLO(
    "D:/GhostTrack-Powered By VINOFlow/GhostTrack/dataset/runs/train_GhostTrack_Demo_8/weights/best.pt"
)  # ใช้ไฟล์ที่เทรนเสร็จ

# ทดสอบโมเดลด้วยข้อมูล val (validation)
model.val(
    data="D:/GhostTrack-Powered By VINOFlow/GhostTrack/dataset/data_check.yaml"
)  # ต้องแน่ใจว่า path ของ data.yaml ถูกต้อ
