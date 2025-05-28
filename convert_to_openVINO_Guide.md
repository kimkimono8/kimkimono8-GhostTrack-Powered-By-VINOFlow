### 🔧 1. กระบวนการที่แนะนำโดย Intel

✅ เริ่มจาก YOLOv8 (.pt) → .onnx

```python
yolo export model=best.pt format=onnx opset=11 dynamic=True simplify=True
```
💡 ใช้ opset=11 หรือสูงกว่า
💡 dynamic=True → รองรับ input size ที่เปลี่ยนได้ (ถ้าใช้ OpenVINO ต้องเปิดไว้)
💡 simplify=True → ลด graph complexity (บางครั้งช่วยให้ convert ได้ง่ายขึ้น)

### 2. แปลงโมเดล ONNX → FP32 หรือ → FP16


```console
ovc yolov8n_ghosttrack.onnx --output_model model_fp32/yolov8n_ghosttrack.xml
ovc yolov8n_ghosttrack.onnx --input "images[1,3,1280,1280]" --output_model model_fp32_fixed/yolov8n_static.xml
```
```python
yolo export model=yolov8n_ghosttrack.pt format=openvino imgsz=1280 simplify=True
```


```console
ovc yolov8n_ghosttrack.onnx --compress_to_fp16 True --output_model output_fp16/yolov8n_ghosttrack.xml
```
```python
yolo export model=yolov8n.pt format=openvino half=True imgsz=1280
```

### 🔧 ตรวจสอบว่า model ใช้ได้ไหม

```python
from openvino.runtime import Core
core = Core()
model = core.read_model("best.xml")
compiled = core.compile_model(model, "GPU")
print("✅ Model ready for GPU")
```

### ✨ Extra: บังคับ shape ตอนแปลง (ถ้า input ไม่ตรง FP16)

```console
mo --input_model best.onnx --data_type FP16 --input_shape "[1,3,640,640]"
```

### ✅ สรุปสาเหตุที่เจอทั้งหมด (พร้อมทางป้องกัน)
ปัญหา	สาเหตุ	วิธีป้องกัน

กรอบไม่ออก	โมเดล export แบบ raw output → ยังไม่ได้ decode (ค่าเกิน, class_id ผิด, score พัง)	ใช้ yolo export format=openvino simplify=True
หรือถ้าจำเป็นต้องใช้ raw → เขียน process_output() เองแบบที่ทำ
class_id = 48	ไม่ใช้ argmax + ไม่มี sigmoid → ได้ logit แทน label	ต้อง decode เอง (sigmoid + argmax) ก่อนใช้
score > 1 / = 40+	ไม่ผ่าน sigmoid → เป็น raw logit	ใช้ sigmoid() กับทั้ง obj_conf และ class_conf
พิกัด box ไม่ตรงภาพจริง	ยังเป็น normalized หรืออยู่ใน grid feature map	ต้อง reverse_letterbox() โดยใช้ scale, pad_left, pad_top จาก preprocess_image()
กรอบซ้อนกันเยอะ	ไม่ใช้ NMS	ต้องใช้ cv2.dnn.NMSBoxes() หรือเขียน NMS เอง
debug ยากมาก	output รูปแบบเปลี่ยนไปตาม version	ต้องดู output.shape + value range ทุกครั้งหลัง export
