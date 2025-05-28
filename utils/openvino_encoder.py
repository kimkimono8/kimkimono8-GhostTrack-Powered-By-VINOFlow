import numpy as np
import cv2
from openvino.runtime import Core


class OpenVINOEncoder:
    def __init__(self, model_path: str, device: str = "GPU"):
        print(f"[INFO] Loading OpenVINO model: {model_path} on device: {device}")
        self.core = Core()

        model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(model, device)

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        print("[✅] Model loaded successfully")
        print("[INFO] Input shape:", self.input_layer.shape)
        print("[INFO] Output shape:", self.output_layer.shape)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Resize to 128x256 and normalize
        resized = cv2.resize(image, (128, 256))
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 256, 128)
        return chw

    def encode(self, image: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(image)
        output = self.compiled_model([input_tensor])[self.output_layer]
        feature = output[0]

        # L2 normalize
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm

        return feature


# ---------- Test block ---------- #
if __name__ == "__main__":
    import os

    MODEL_XML = "person-reidentification-retail-0277.xml"  # แก้ชื่อไฟล์ตรงนี้ถ้าไม่ตรง
    if not os.path.exists(MODEL_XML):
        print(f"[❌] Model not found: {MODEL_XML}")
        exit()

    encoder = OpenVINOEncoder(MODEL_XML, device="GPU")

    dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    feat = encoder.encode(dummy_img)

    print("[✅] Feature vector shape:", feat.shape)
    print("[✅] Feature vector (first 5):", feat[:5])
    print("[✅] L2 Norm:", np.linalg.norm(feat))
