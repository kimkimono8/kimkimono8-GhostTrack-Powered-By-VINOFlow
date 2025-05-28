import os
import numpy as np
import cv2
from ultralytics import YOLO
import json
from openvino import Core

DEVICE = "GPU"


with open(
    "D:/GhostTrack-Powered By VINOFlow/GhostTrack/dataset/runs/train_GhostTrack_Clean/results_detailed.json"
) as f:
    CLASS_NAMES = json.load(f)["class_names"]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def decode_boxes(prediction):
    boxes = prediction[..., :4]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return boxes_xyxy


def calculate_iou(box1, boxes):
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = boxes.T
    inter_x1 = np.maximum(x1, x1_b)
    inter_y1 = np.maximum(y1, y1_b)
    inter_x2 = np.minimum(x2, x2_b)
    inter_y2 = np.minimum(y2, y2_b)
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    box_area = (x2 - x1) * (y2 - y1)
    boxes_area = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = box_area + boxes_area - inter_area
    return inter_area / union_area


def nms(predictions, conf_threshold=0.5, nms_threshold=0.4):
    mask = predictions[..., 4] > conf_threshold
    predictions = predictions[mask]

    if len(predictions) == 0:
        return [], []

    scores = predictions[..., 4] * predictions[..., 5:].max(-1)
    sorted_indices = np.argsort(scores)[::-1]
    predictions = predictions[sorted_indices]

    boxes, final_scores = [], []

    while len(predictions) > 0:
        box = predictions[0]
        boxes.append(box[:4])
        final_scores.append(scores[0])
        iou = calculate_iou(box[0:4], predictions[1:, 0:4])
        predictions = predictions[1:][iou < nms_threshold]

    return np.array(boxes), np.array(final_scores)


def process_output(output, conf_threshold=0.3, iou_threshold=0.45):
    output = output.squeeze().transpose(1, 0)  # (33600, 8)
    boxes = output[:, :4]
    objectness = sigmoid(output[:, 4:5])
    num_classes = len(CLASS_NAMES)
    class_logits = output[:, 5 : 5 + num_classes]
    class_scores = sigmoid(class_logits)

    class_ids = np.argmax(class_scores, axis=1)
    class_conf = np.max(class_scores, axis=1, keepdims=True)
    scores = objectness * class_conf

    mask = scores[:, 0] > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # xywh → xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores[:, 0].tolist(), conf_threshold, iou_threshold
    )
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])

    indices = indices.flatten()
    boxes_xyxy = boxes_xyxy[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]

    # ✅ move print here
    print(f"🟢 พบวัตถุทั้งหมด {len(class_ids)} รายการ")
    for i, cid in enumerate(class_ids):
        print(f"  - {CLASS_NAMES[cid]} ({float(scores[i]):.2f})")

    return boxes_xyxy, scores.flatten(), class_ids


def reverse_letterbox(boxes, scale, pad_left, pad_top):
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, [0, 2]] /= scale
    boxes[:, [1, 3]] /= scale
    return boxes


def draw_boxes(image, boxes, scores, class_ids):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = class_ids[i]
        label = (
            f"{CLASS_NAMES[class_id]} ({scores[i]:.2f})"
            if class_id < len(CLASS_NAMES)
            else f"ID:{class_id}"
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )


def preprocess_image(frame, target_size=(1280, 1280)):
    h0, w0 = frame.shape[:2]
    scale = min(target_size[0] / h0, target_size[1] / w0)
    nh, nw = int(h0 * scale), int(w0 * scale)
    image_resized = cv2.resize(frame, (nw, nh))
    pad_top = (target_size[0] - nh) // 2
    pad_left = (target_size[1] - nw) // 2
    image_padded = cv2.copyMakeBorder(
        image_resized,
        pad_top,
        target_size[0] - nh - pad_top,
        pad_left,
        target_size[1] - nw - pad_left,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = np.transpose(image_rgb, (2, 0, 1))[np.newaxis, ...]
    return input_tensor, frame, scale, pad_left, pad_top


def infer_and_postprocess(model_path, input_video_path, output_video_path):

    # === กำหนดค่า Environment Variables สำหรับการใช้งาน CPU ===
    os.environ["OMP_NUM_THREADS"] = "8"  # กำหนดจำนวน Thread ที่จะใช้
    os.environ["OPENVINO_CPU_THREADS_NUM"] = "8"  # CPU Thread จำนวนที่เหมาะสม
    os.environ["OPENVINO_CPU_BIND_THREAD"] = "YES"  # Bind Thread กับ Core
    os.environ["OPENVINO_CPU_THROUGHPUT_STREAMS"] = "2"  # ให้ใช้ Stream เดียว

    core = Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, DEVICE)
    print("[✅] Model loaded successfully")
    print("[INFO] Input shape:", compiled_model)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, original_image, scale, pad_left, pad_top = preprocess_image(frame)
        output_layer = compiled_model.output(0)
        result = compiled_model([input_tensor])
        output = result[output_layer]  # shape: (1, N, 6)
        print("Final output shape:", output.shape)
        print("raw class scores:", output[0, 5:, 0])  # class logits จาก anchor 0
        print("expected CLASS_NAMES length =", len(CLASS_NAMES))

        # ✅ Process output
        boxes, scores, class_ids = process_output(output)
        boxes = reverse_letterbox(boxes, scale, pad_left, pad_top)

        if len(boxes) > 0:
            draw_boxes(original_image, boxes, scores, class_ids)
        out.write(original_image)
        cv2.imshow("Result", original_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ======== เรียกใช้งาน =========
input_video = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/debug_video.mp4"
output_video = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/inference_result_video.avi"
model_path = "D:/GhostTrack-Powered By VINOFlow/GhostTrack/best.xml"

infer_and_postprocess(model_path, input_video, output_video)
