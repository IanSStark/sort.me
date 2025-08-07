# capture.py
"""
Capture a single RGB frame from Pi Camera 2 and return it as a NumPy array.
Requires: python3-picamera2, numpy
"""
from picamera2 import Picamera2, Preview
import time


def capture_frame(resolution=(1920, 1080), preview=False):
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": resolution},
        lores={"size": (640, 480)},
        display="lores"
    )
    picam2.configure(config)

    if preview:
        picam2.start_preview(Preview.QTGL)  # optional window
    picam2.start()
    time.sleep(2)                          # camera warm-up
    frame = picam2.capture_array()
    if preview:
        picam2.stop_preview()
    picam2.close()
    return frame


if __name__ == "__main__":
    # Simple verification: capture and write to disk
    import cv2
    img = capture_frame()
    cv2.imwrite("test_capture.jpg", img)
    print("Wrote test_capture.jpg")
