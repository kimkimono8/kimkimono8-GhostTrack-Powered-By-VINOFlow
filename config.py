# analyzer/config.py
import os
from pathlib import Path
import yaml


RTSP_URL = "rtsp://admin:Kimono1995@192.168.1.130:554/h264/ch01/main/av_stream?tcp"

# NAS PATH
NAS_DIR = Path(r"S:/กล้องหน้าบ้าน")

# --- Base Path ---
BASE_DIR = Path("D:/Tflite")
MODEL_BASE_DIR = Path("D:/Tflite/model")
HASS_DIR = Path("H:/www/output")
LOG_DIR = BASE_DIR / "logs"

# --- Log Files ---
APP_LOG_FILE = LOG_DIR / "app.log"
UTILS_LOG_FILE = LOG_DIR / "utils.log"
LOG_ANALYZE_PATH = LOG_DIR / "analyze.log"
LOG_FILE_PATH = LOG_DIR / "log.txt"

# --- Folder Paths ---
INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"


# --- Model Folders Paths ---
REID_MODEL_PATH = MODEL_BASE_DIR / "REID_OV"

# --- Model Paths ---
MODEL_PATH = MODEL_DIR / "yolov8n_ghosttrack_fp16.xml"

REID_MODEL_PATH = MODEL_BASE_DIR / "person-reidentification-retail-0277.xml"

# สำหรับกรองหลัง sigmoid (ก่อนส่ง NMS)
CONF_THRES = 0.25


DEBUG = False  # ✅ สลับเปิด/ปิด imshow
LOCAL_IP = "192.18.1.20"  # หรือ ip เครื่อง เช่น "192.168.1.10"


# Class Labels
def load_class_labels_from_yaml(path="D:/Tflite/model/metadata.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["names"].items()}


CLASS_LABELS = load_class_labels_from_yaml()

TARGET_CLASSES = None  # {"person", "car", "cat"}
UNKNOWN_CLASS_DIR = os.path.join(DATASET_DIR, "Unknown")

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "7720396531:AAGoPJ_BI9gqfR9YvlHnjuFsJM39JXslmZs"
)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7843062361")
LOG_ANALYZE_PATH = "logs/analyze.log"

# Video Constraints
MIN_DURATION = 5
MAX_DURATION = 65


INTERVAL = 5  # frame interval for detection

# File Names
STANDARD_VIDEO_NAME = "clip.mp4"
SNAP_FILENAME = "snap.jpg"

LOCAL_IP = "192.168.1.20"
