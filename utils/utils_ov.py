import cv2
import numpy as np
from collections import defaultdict
from config import CLASS_LABELS, TARGET_CLASSES, CONF_THRES


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding."""
    shape = image.shape[:2]  # h, w
    ratio = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape[0] - new_unpad[0]) / 2
    dh = (new_shape[1] - new_unpad[1]) / 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, dw, dh


def preprocess_yolo(image):
    """Prepares image for YOLOv8 inference"""
    img, ratio, dw, dh = letterbox(image, new_shape=(640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img, ratio, dw, dh


def postprocess(output, shape, ratio, dw, dh):
    h_img, w_img = shape[:2]
    preds = np.squeeze(output).T  # ✅ คงที่ ไม่ต้องเช็ค shape

    detections = []
    for row in preds:
        cx, cy, w, h = row[:4]
        obj_conf = sigmoid(row[4])
        class_scores = sigmoid(row[5:])
        cls_id = int(np.argmax(class_scores))
        cls_conf = class_scores[cls_id]
        conf = obj_conf * cls_conf
        if conf < CONF_THRES or cls_id not in CLASS_LABELS:
            continue

        class_name = CLASS_LABELS[cls_id]
        if TARGET_CLASSES and class_name not in TARGET_CLASSES:
            # print(
            #    f"[CLASS FILTER] cls_id={cls_id}, name={class_name}, allowed={TARGET_CLASSES}"
            # )
            continue

        # undo letterbox
        cx = (cx - dw) / ratio
        cy = (cy - dh) / ratio
        w /= ratio
        h /= ratio

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        detections.append(((x1, y1, x2, y2), conf, class_name))

    return detections


def apply_nms(detections, iou_threshold=0.5, conf_threshold=0.25):
    boxes, scores = [], []
    for (x1, y1, x2, y2), conf, _ in detections:
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    if len(indices) == 0:
        return []

    indices = indices.flatten()
    # print(f"[NMS] Before: {len(detections)} → After: {len(indices)}")
    return [detections[i] for i in indices]


def apply_classwise_nms(detections, conf_threshold=0.3, iou_threshold=0.5):
    classwise = defaultdict(list)
    for box, conf, class_name in detections:
        classwise[class_name].append((box, conf, class_name))

    final_dets = []
    for cls, dets in classwise.items():
        boxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), _, _ in dets]
        scores = [float(conf) for _, conf, _ in dets]
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        indices = indices.flatten() if len(indices) > 0 else []
        final_dets.extend([dets[i] for i in indices])
        # print(f"[NMS] {cls}: {len(dets)} → {len(indices)}")

    # print(f"[NMS] Total kept: {len(final_dets)}")
    return final_dets
