import subprocess
import numpy as np
import time
import threading


class RTSPFFmpegReader:
    def __init__(
        self,
        rtsp_url,
        width=640,
        height=480,
        ffmpeg_path="ffmpeg",
        retry_delay=5,
        max_failures=5,
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.ffmpeg_path = ffmpeg_path
        self.retry_delay = retry_delay
        self.max_failures = max_failures
        self.frame_size = width * height * 3
        self.process = None
        self.running = False
        self.thread = None
        self.frame = None
        self.lock = threading.Lock()
        self.failure_count = 0

    def _build_cmd(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-analyzeduration",
            "20000000",
            "-probesize",
            "20000000",
            "-hwaccel",
            "qsv",
            "-hwaccel_output_format",
            "qsv",
            "-c:v",
            "h264_qsv",
            "-i",
            self.rtsp_url,
            "-vf",
            "vpp_qsv=w=854:h=480",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        return cmd

    def _reader_loop(self):
        while self.running:
            if self.process is None or self.process.poll() is not None:
                self._start_process()
                if not self.process or self.process.poll() is not None:
                    print("[⚠️] Failed to start FFmpeg. Retrying...")
                    self.failure_count += 1
                    if self.failure_count >= self.max_failures:
                        print("[❌] Max failures reached. Giving up.")
                        self.running = False
                        return
                    time.sleep(self.retry_delay)
                    continue

            raw = self.process.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                print("[⛔] Incomplete frame. Restarting FFmpeg...")
                self._stop_process()
                self.failure_count += 1
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            with self.lock:
                self.frame = frame.copy()
            self.failure_count = 0

    def _start_process(self):
        self._stop_process()
        try:
            cmd = self._build_cmd()
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8,
            )
            print("[▶️] FFmpeg started")
        except Exception as e:
            print(f"[❌] Failed to start FFmpeg: {e}")
            self.process = None

    def _stop_process(self):
        if self.process:
            self.process.terminate()
            self.process = None
            print("[🛑] FFmpeg stopped")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self._stop_process()
        if self.thread:
            self.thread.join()
        print("[🚪] RTSP reader stopped")
