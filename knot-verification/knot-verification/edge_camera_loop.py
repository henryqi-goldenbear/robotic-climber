"""
Continuous inference loop for a Jetson / Raspberry Pi with a USB or CSI
camera. Runs the full detect -> extract -> classify pipeline on a live
feed and overlays the verdict on the video window.

Swap `cv2.VideoCapture(0)` for a Jetson CSI camera GStreamer pipeline if
you're on a Jetson with a ribbon-cable camera instead of USB.
"""
import time

import cv2
from PIL import Image

from pipeline import KnotInspectionPipeline

INFER_EVERY_N_FRAMES = 5   # throttle inference so it doesn't fight the camera FPS


def main():
    pipeline = KnotInspectionPipeline()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check the device index / connection.")

    frame_idx = 0
    last_result = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % INFER_EVERY_N_FRAMES == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                t0 = time.time()
                last_result = pipeline.predict(image)
                latency_ms = (time.time() - t0) * 1000
                if last_result.label:
                    print(f"[{latency_ms:.0f} ms] {last_result.label} "
                          f"(conf {last_result.confidence:.2%})")
                else:
                    print(f"[{latency_ms:.0f} ms] no knot detected")

            if last_result and last_result.box:
                x1, y1, x2, y2 = map(int, last_result.box)
                color = (0, 200, 0) if last_result.label == "Correct" else (0, 0, 220)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, last_result.label or "", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Knot Inspection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
