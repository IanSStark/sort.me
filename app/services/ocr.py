# app/services/ocr.py
import logging
from pathlib import Path
from typing import Dict, Any
import pytesseract
from PIL import Image

logger = logging.getLogger("ocr")

class OCREngine:
    def __init__(self, lang: str = "eng", psm: int = 6):
        """
        Initialize OCR engine.
        :param lang: Language for Tesseract (default: English)
        :param psm: Page segmentation mode (default: 6 - assume block of text)
        """
        self.lang = lang
        self.psm = psm
        self._config = f"--oem 3 --psm {self.psm}"
        logger.info(f"OCR initialized: engine=tesseract, lang={self.lang}, psm={self.psm}")

    def process(self, image_path: str | Path) -> Dict[str, Any]:
        """
        Perform OCR on the entire image from top-left to bottom-right.
        """
        image_path = Path(image_path)
        logger.info(f"OCR started on {image_path}")

        try:
            # Open image
            img = Image.open(image_path)

            # Run OCR on the whole image
            raw_text = pytesseract.image_to_string(
                img,
                lang=self.lang,
                config=self._config
            )

            # Get word-level bounding box and confidence
            details = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=self._config,
                output_type=pytesseract.Output.DICT
            )

            logger.info(f"OCR completed: {len(raw_text.strip())} characters extracted")

            return {
                "image": str(image_path),
                "text": raw_text.strip(),
                "words": [
                    {
                        "text": details["text"][i],
                        "conf": int(details["conf"][i]) if details["conf"][i] != "-1" else None,
                        "bbox": (
                            details["left"][i],
                            details["top"][i],
                            details["width"][i],
                            details["height"][i]
                        )
                    }
                    for i in range(len(details["text"])) if details["text"][i].strip()
                ]
            }

        except Exception as e:
            logger.error(f"OCR failed on {image_path}: {e}")
            return {
                "image": str(image_path),
                "text": "",
                "error": str(e),
                "words": []
            }


# Global instance
ocr_engine = OCREngine()
