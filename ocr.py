# ocr.py
"""
Extract the card title from a fixed region of interest (ROI) in the input image.
Requires: opencv-python, pytesseract, Pillow, numpy
"""

from typing import Tuple

import cv2
import numpy as np
import pytesseract


# ----- CONFIGURABLE CONSTANTS -----
# Pixel rectangle where the card title appears: (x, y, width, height)
ROI: Tuple[int, int, int, int] = (100, 150, 600, 100)

# Tesseract configuration: OCR Engine Mode 3 (LSTM only), Page Seg Mode 6 (single line)
TESSERACT_CFG = (
    "--oem 3 --psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
)


# ----- IMPLEMENTATION -----
def _preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Automatic Otsu threshold for high contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_title(full_img: np.ndarray, roi: Tuple[int, int, int, int] = ROI) -> str:
    x, y, w, h = roi
    roi_img = full_img[y : y + h, x : x + w]
    processed = _preprocess(roi_img)
    text = pytesseract.image_to_string(processed, config=TESSERACT_CFG)
    return text.strip()


if __name__ == "__main__":
    from capture import capture_frame

    frame = capture_frame(preview=False)
    title = extract_title(frame)
    print("Detected title:", repr(title))

