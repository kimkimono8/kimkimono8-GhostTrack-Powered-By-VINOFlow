from utils.rtsp_reader import RTSPFFmpegReader
from utils.inference_async import InferencePipeline
from config import RTSP_URL
from utils.web_stream import start_flask


def main():
    rtsp = RTSPFFmpegReader(RTSP_URL, width=854, height=480)
    pipeline = InferencePipeline()

    start_flask()
    rtsp.start()
    pipeline.start()

    try:
        while True:
            frame = rtsp.read()
            if frame is not None:
                pipeline.enqueue_frame(frame)
                pipeline.render_if_ready()

    except KeyboardInterrupt:
        print("[🚪] Exiting...")
    finally:
        rtsp.stop()
        pipeline.stop()


if __name__ == "__main__":
    main()
