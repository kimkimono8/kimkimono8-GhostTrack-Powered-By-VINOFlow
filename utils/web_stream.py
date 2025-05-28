# web_stream.py
from flask import Flask, Response
import threading
import cv2
from config import LOCAL_IP

app = Flask(__name__)
latest_frame = None
lock = threading.Lock()


def update_frame(frame):
    global latest_frame
    with lock:
        latest_frame = frame.copy()


def generate():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            ret, jpeg = cv2.imencode(".jpg", latest_frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start_flask():
    threading.Thread(
        target=lambda: app.run(
            host=LOCAL_IP, port=5005, debug=False, use_reloader=False
        ),
        daemon=True,
    ).start()
