from ultralytics import YOLO

model = YOLO("D:/GhostTrack-Powered By VINOFlow/GhostTrack/best.pt")

print("✅ Class names:", model.names)
print("✅ จำนวนคลาส:", len(model.names))

# ✅ ตรวจสอบจาก Detect head โดยตรง
detect_head = model.model.model[-1]
print("✅ Detect head type:", type(detect_head))
print("✅ จำนวนคลาสจาก head:", detect_head.nc)

# ✅ ตรวจสอบจำนวน anchor outputs
print("✅ Output channels (รวม class + 5):", detect_head.no)  # no = nc + 5
