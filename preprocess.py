from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from utils import save_debug_image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    # CLAHE improves local contrast for faint handwriting and uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def estimate_skew_angle(binary: np.ndarray) -> float:
    inv = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(inv)
    if coords is None:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # minAreaRect angle normalization.
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def deskew(binary: np.ndarray, angle_threshold: float = 0.2) -> Tuple[np.ndarray, float]:
    angle = estimate_skew_angle(binary)
    if abs(angle) < angle_threshold:
        return binary, 0.0

    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def upscale_for_ocr(image: np.ndarray, min_width: int = 1800) -> np.ndarray:
    h, w = image.shape[:2]
    if w >= min_width:
        return image

    scale = min_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def preprocess_image(
    image: np.ndarray,
    debug_dir: str | Path | None = None,
    prefix: str = "img",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply a robust preprocessing chain for OCR."""
    metrics: Dict[str, float] = {}

    gray = to_grayscale(image)
    contrast = enhance_contrast(gray)
    clean = denoise(contrast)
    binary = adaptive_binarize(clean)
    deskwed, angle = deskew(binary)
    upscaled = upscale_for_ocr(deskwed)

    metrics["estimated_skew_angle"] = angle
    metrics["input_width"] = float(image.shape[1])
    metrics["output_width"] = float(upscaled.shape[1])

    if debug_dir:
        save_debug_image(gray, debug_dir, f"{prefix}_1_gray.png")
        save_debug_image(contrast, debug_dir, f"{prefix}_2_contrast.png")
        save_debug_image(clean, debug_dir, f"{prefix}_3_denoise.png")
        save_debug_image(binary, debug_dir, f"{prefix}_4_binary.png")
        save_debug_image(deskwed, debug_dir, f"{prefix}_5_deskew.png")
        save_debug_image(upscaled, debug_dir, f"{prefix}_6_upscaled.png")

    return upscaled, metrics
