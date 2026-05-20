from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from utils import normalize_ocr_text


@dataclass
class OCRResult:
    raw_text: str
    cleaned_text: str
    confidence: float
    words_detected: int
    warnings: List[str]


def _build_tesseract_config(psm: int = 6) -> str:
    whitelist = "KQRBNOabcdefgh12345678xX=+#-./:()!?Oo "
    return f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"


def _compute_confidence(data: Dict[str, List[str]]) -> float:
    conf_values: List[float] = []
    for c in data.get("conf", []):
        try:
            value = float(c)
        except Exception:
            continue
        if value >= 0:
            conf_values.append(value)
    if not conf_values:
        return 0.0
    return float(sum(conf_values) / len(conf_values))


def extract_text_from_image(image: np.ndarray, psm: int = 6) -> OCRResult:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    config = _build_tesseract_config(psm=psm)
    warnings: List[str] = []

    try:
        raw_text = pytesseract.image_to_string(gray, config=config)
        data = pytesseract.image_to_data(gray, config=config, output_type=Output.DICT)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract executable not found. Install Tesseract and add it to PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OCR extraction failed: {exc}") from exc

    conf = _compute_confidence(data)
    cleaned = normalize_ocr_text(raw_text)

    if conf < 45:
        warnings.append("Low OCR confidence detected; move correction may be needed.")
    if not cleaned.strip():
        warnings.append("OCR produced empty text after normalization.")

    words_detected = len([t for t in data.get("text", []) if str(t).strip()])

    return OCRResult(
        raw_text=raw_text,
        cleaned_text=cleaned,
        confidence=conf,
        words_detected=words_detected,
        warnings=warnings,
    )
