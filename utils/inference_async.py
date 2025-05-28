import cv2
import numpy as np
from queue import Queue
from openvino.runtime import Core, AsyncInferQueue
from utils.openvino_encoder import OpenVINOEncoder
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from utils.utils_ov import preprocess_yolo, postprocess, apply_classwise_nms
from utils.web_stream import update_frame
from config import MODEL_PATH, REID_MODEL_PATH, DEBUG, CLASS_LABELS


class InferencePipeline:
    def __init__(self):
        self.queue = Queue(maxsize=5)
        self.results = Queue()
        self.ie = Core()

        # YOLOv8
        model = self.ie.read_model(MODEL_PATH)
        self.compiled_model = self.ie.compile_model(model, "GPU")
        self.infer_queue = AsyncInferQueue(self.compiled_model, 2)
        self.infer_queue.set_callback(self.callback)

        self.input_tensor = self.compiled_model.input(0)

        # REID
        self.encoder = OpenVINOEncoder(REID_MODEL_PATH, device="GPU")

        # DeepSort
        metric = NearestNeighborDistanceMetric(
            "cosine", matching_threshold=0.5, budget=100
        )
        self.tracker = Tracker(metric)

    def start(self):
        pass  # placeholder

    def stop(self):
        pass  # placeholder

    def enqueue_frame(self, frame):
        if not self.queue.full():
            input_tensor, ratio, dw, dh = preprocess_yolo(frame)
            shape = frame.shape
            self.infer_queue.start_async(
                inputs={self.input_tensor: input_tensor},
                userdata=(frame.copy(), shape, ratio, dw, dh),
            )

    def callback(self, request, userdata):
        frame, shape, ratio, dw, dh = userdata
        output = request.get_output_tensor().data
        dets = postprocess(output, shape, ratio, dw, dh)
        dets = apply_classwise_nms(dets)

        detections = []
        for (x1, y1, x2, y2), conf, class_name in dets:
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 64 or crop.shape[1] < 64:
                continue
            feature = self.encoder.encode(crop)
            tlwh = [x1, y1, x2 - x1, y2 - y1]
            detections.append(Detection(tlwh, conf, feature))

        self.tracker.predict()
        self.tracker.update(detections)

        self.results.put((frame, self.tracker.tracks))

    def render_if_ready(self):
        if not self.results.empty():
            frame, tracks = self.results.get()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                track_id = track.track_id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            update_frame(frame)

            if DEBUG:
                cv2.imshow("GhostTrack:Powered by VINOFlow Async", frame)
                cv2.waitKey(1)
