from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from utils import save_debug_image


@dataclass
class PreprocessResult:
    resized_color: np.ndarray
    gray: np.ndarray
    contrast: np.ndarray
    denoised: np.ndarray
    sharpened: np.ndarray
    thresholded: np.ndarray
    horizontal_lines: np.ndarray
    vertical_lines: np.ndarray
    line_mask: np.ndarray
    table_bbox: Tuple[int, int, int, int]
    table_color: np.ndarray
    table_gray: np.ndarray
    table_thresholded: np.ndarray
    table_cleaned: np.ndarray
    metrics: Dict[str, float]
    warnings: List[str]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)


def sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    return cv2.filter2D(gray, -1, kernel)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def resize_for_ocr(image: np.ndarray, min_width: int = 2400) -> np.ndarray:
    height, width = image.shape[:2]
    if width >= min_width:
        return image

    scale = min_width / float(width)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
    return resized


def _find_contours(mask: np.ndarray):
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 2:
        contours, _ = contours_info
    else:
        _, contours, _ = contours_info
    return contours


def detect_table_bbox(line_mask: np.ndarray, image_shape: Tuple[int, int], warnings: List[str]) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    page_area = float(height * width)

    contours = _find_contours(line_mask)
    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(w * h)
        if bbox_area < page_area * 0.03:
            continue
        candidates.append((bbox_area, (x, y, w, h)))

    if candidates:
        _, bbox = max(candidates, key=lambda item: item[0])
        x, y, w, h = bbox
        pad_x = max(12, int(w * 0.02))
        pad_y = max(12, int(h * 0.02))
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(width - x, w + pad_x * 2)
        h = min(height - y, h + pad_y * 2)
        return x, y, w, h

    warnings.append("Could not confidently localize the scoresheet table; using a central fallback crop.")
    x = int(width * 0.05)
    y = int(height * 0.05)
    w = int(width * 0.90)
    h = int(height * 0.90)
    return x, y, w, h


def crop_region(image: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
    x, y, w, h = bbox
    height, width = image.shape[:2]
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(width, x + w + pad)
    bottom = min(height, y + h + pad)
    return image[top:bottom, left:right].copy()


def detect_table_lines(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverted = cv2.bitwise_not(binary)
    height, width = binary.shape[:2]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, width // 45), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, height // 40)))

    horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)

    line_mask = cv2.bitwise_or(horizontal, vertical)
    return horizontal, vertical, line_mask


def remove_table_lines(binary: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    cleaned = binary.copy()
    cleaned[line_mask > 0] = 255
    return cleaned


def preprocess_image(
    image: np.ndarray,
    debug_dir: str | Path | None = None,
    prefix: str = "img",
) -> PreprocessResult:
    warnings: List[str] = []
    metrics: Dict[str, float] = {}

    resized_color = resize_for_ocr(image)
    gray = to_grayscale(resized_color)
    contrast = enhance_contrast(gray)
    denoised = denoise(contrast)
    sharpened = sharpen(denoised)
    thresholded = adaptive_binarize(sharpened)

    horizontal_lines, vertical_lines, line_mask = detect_table_lines(thresholded)
    table_bbox = detect_table_bbox(line_mask, thresholded.shape, warnings)

    table_color = crop_region(resized_color, table_bbox, pad=2)
    table_gray = crop_region(sharpened, table_bbox, pad=2)
    table_thresholded = crop_region(thresholded, table_bbox, pad=2)
    table_line_mask = crop_region(line_mask, table_bbox, pad=2)
    table_cleaned = remove_table_lines(table_thresholded, table_line_mask)

    metrics["input_width"] = float(image.shape[1])
    metrics["input_height"] = float(image.shape[0])
    metrics["resized_width"] = float(resized_color.shape[1])
    metrics["resized_height"] = float(resized_color.shape[0])
    metrics["table_width"] = float(table_gray.shape[1]) if table_gray.size else 0.0
    metrics["table_height"] = float(table_gray.shape[0]) if table_gray.size else 0.0
    metrics["table_x"] = float(table_bbox[0])
    metrics["table_y"] = float(table_bbox[1])

    if debug_dir:
        save_debug_image(gray, debug_dir, f"{prefix}_1_gray.png")
        save_debug_image(contrast, debug_dir, f"{prefix}_2_contrast.png")
        save_debug_image(denoised, debug_dir, f"{prefix}_3_denoised.png")
        save_debug_image(sharpened, debug_dir, f"{prefix}_4_sharpened.png")
        save_debug_image(thresholded, debug_dir, f"{prefix}_5_thresholded.png")
        save_debug_image(line_mask, debug_dir, f"{prefix}_6_line_mask.png")
        save_debug_image(table_gray, debug_dir, f"{prefix}_7_table_crop.png")
        save_debug_image(table_thresholded, debug_dir, f"{prefix}_8_table_thresholded.png")
        save_debug_image(table_cleaned, debug_dir, f"{prefix}_9_table_lines_removed.png")

    return PreprocessResult(
        resized_color=resized_color,
        gray=gray,
        contrast=contrast,
        denoised=denoised,
        sharpened=sharpened,
        thresholded=thresholded,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        line_mask=line_mask,
        table_bbox=table_bbox,
        table_color=table_color,
        table_gray=table_gray,
        table_thresholded=table_thresholded,
        table_cleaned=table_cleaned,
        metrics=metrics,
        warnings=warnings,
    )
