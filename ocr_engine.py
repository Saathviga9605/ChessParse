from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import regex as re

from preprocess import PreprocessResult, crop_region
from utils import ensure_dir, normalize_ocr_text


@dataclass
class OCRRowResult:
    row_index: int
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    raw_text: str
    psm: int


@dataclass
class OCRTableResult:
    rows: List[OCRRowResult]
    assembled_text: str
    average_confidence: float
    warnings: List[str]
    visualization_path: Optional[str]
    table_bbox: Tuple[int, int, int, int]


@lru_cache(maxsize=1)
def _get_paddle_ocr():
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install 'paddleocr' and 'paddlepaddle' in the active environment."
        ) from exc

    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
        det=True,
        rec=True,
    )


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _blue_ink_enhancement(color_image: np.ndarray) -> np.ndarray:
    if color_image.size == 0 or len(color_image.shape) != 3:
        return np.array([], dtype=np.uint8)

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (78, 16, 18), (145, 255, 255))

    blue, green, red = cv2.split(color_image)
    dominance = blue.astype(np.int16) - ((green.astype(np.int16) + red.astype(np.int16)) // 2)
    dominance = np.clip(dominance, 0, 255).astype(np.uint8)
    dominance = cv2.GaussianBlur(dominance, (3, 3), 0)

    ink = np.full_like(dominance, 255)
    ink_mask = blue_mask > 0
    ink[ink_mask] = 255 - dominance[ink_mask]
    ink = cv2.medianBlur(ink, 3)
    return ink


def _trim_inner_border(image: np.ndarray, border: int = 12) -> np.ndarray:
    if image.size == 0:
        return image
    height, width = image.shape[:2]
    if height <= border * 2 or width <= border * 2:
        return image.copy()
    return image[border : height - border, border : width - border].copy()


def _resize_for_row_ocr(image: np.ndarray, scale: float = 3.0) -> np.ndarray:
    if image.size == 0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _clahe_and_denoise(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    return gray


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )


def _remove_gentle_grid_lines(binary: np.ndarray) -> np.ndarray:
    if binary.size == 0:
        return binary

    inverted = cv2.bitwise_not(binary)
    height, width = binary.shape[:2]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, width // 60), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height // 60)))

    horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)
    line_mask = cv2.bitwise_or(horizontal, vertical)

    cleaned = binary.copy()
    cleaned[line_mask > 0] = 255
    return cleaned


def _group_consecutive(indices: np.ndarray) -> List[np.ndarray]:
    if indices.size == 0:
        return []

    groups: List[np.ndarray] = []
    start = 0
    for idx in range(1, len(indices)):
        if indices[idx] != indices[idx - 1] + 1:
            groups.append(indices[start:idx])
            start = idx
    groups.append(indices[start:])
    return groups


def _merge_close_positions(positions: Sequence[int], tolerance: int = 8) -> List[int]:
    if not positions:
        return []

    sorted_positions = sorted(int(pos) for pos in positions)
    merged: List[int] = [sorted_positions[0]]
    cluster: List[int] = [sorted_positions[0]]

    for position in sorted_positions[1:]:
        if position - cluster[-1] <= tolerance:
            cluster.append(position)
        else:
            merged[-1] = int(round(sum(cluster) / len(cluster)))
            cluster = [position]
            merged.append(position)

    merged[-1] = int(round(sum(cluster) / len(cluster)))
    return merged


def _detect_line_centers(mask: np.ndarray, orientation: str, threshold_ratio: float = 0.3) -> List[int]:
    if mask.size == 0:
        return []

    if orientation == "horizontal":
        projection = np.sum(mask > 0, axis=1)
    else:
        projection = np.sum(mask > 0, axis=0)

    max_value = int(projection.max()) if projection.size else 0
    if max_value <= 0:
        return []

    threshold = max(3, int(max_value * threshold_ratio))
    indices = np.where(projection >= threshold)[0]
    groups = _group_consecutive(indices)

    centers: List[int] = []
    for group in groups:
        if group.size == 0:
            continue
        centers.append(int(round(float(np.mean(group)))))

    return _merge_close_positions(centers, tolerance=8)


def _build_intervals(boundary_centers: Sequence[int], size: int, min_span: int, margin: int = 3) -> List[Tuple[int, int]]:
    centers = [int(center) for center in boundary_centers if 0 < int(center) < size]
    boundaries = [0] + sorted(set(centers)) + [size]

    intervals: List[Tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        left = max(0, start + margin)
        right = min(size, end - margin)
        if right - left >= min_span:
            intervals.append((left, right))
    return intervals


def _equal_intervals(size: int, parts: int) -> List[Tuple[int, int]]:
    parts = max(1, parts)
    step = size / float(parts)
    intervals: List[Tuple[int, int]] = []
    for idx in range(parts):
        left = int(round(idx * step))
        right = int(round((idx + 1) * step))
        if right > left:
            intervals.append((left, right))
    return intervals


def _normalize_move_like_text(text: str) -> str:
    if not text:
        return text

    cleaned = normalize_ocr_text(text)
    cleaned = cleaned.replace("×", "x")
    cleaned = re.sub(r"\b0-0-0\b", "O-O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b0-0\b", "O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bo-o-o\b", "O-O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bo-o\b", "O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b0o\b", "O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\boo\b", "O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([KQRBN])([a-h])([sS])\b", lambda m: f"{m.group(1).upper()}{m.group(2)}5", cleaned)
    cleaned = re.sub(r"\bBbs\b", "Bb5", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bBbS\b", "Bb5", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([a-h])([sS])\b", lambda m: f"{m.group(1)}5", cleaned)
    cleaned = re.sub(r"\b([a-h])([lI])\b", lambda m: f"{m.group(1)}1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_row_ocr_inputs(row_color: np.ndarray, row_gray: np.ndarray) -> List[np.ndarray]:
    variants: List[np.ndarray] = []

    if row_color is not None and row_color.size > 0 and len(row_color.shape) == 3:
        blue = _blue_ink_enhancement(row_color)
        if blue.size > 0:
            blue = _trim_inner_border(blue, border=12)
            blue = _clahe_and_denoise(blue)
            blue = _resize_for_row_ocr(blue, scale=3.0)
            if blue.size > 0:
                blue = _adaptive_threshold(blue)
                blue = _remove_gentle_grid_lines(blue)
                variants.append(blue)

    gray = _ensure_gray(row_gray)
    if gray.size > 0:
        if float(np.mean(gray)) < 127.0:
            gray = cv2.bitwise_not(gray)
        gray = _trim_inner_border(gray, border=12)
        gray = _clahe_and_denoise(gray)
        gray = _resize_for_row_ocr(gray, scale=3.0)
        if gray.size > 0:
            gray = _adaptive_threshold(gray)
            gray = _remove_gentle_grid_lines(gray)
            variants.append(gray)

    return variants


def _ocr_with_paddle(image: np.ndarray) -> Tuple[str, float, int, List[Tuple[str, float]]]:
    engine = _get_paddle_ocr()
    try:
        result = engine.ocr(image, cls=True)
    except Exception:
        return "", 0.0, 0, []

    if not result:
        return "", 0.0, 0, []

    lines = result[0] if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list) else result
    parsed_lines: List[Tuple[float, float, str, float]] = []

    for line in lines:
        if not line or len(line) < 2:
            continue
        box = line[0]
        text_conf = line[1]
        if not text_conf or len(text_conf) < 2:
            continue
        text = str(text_conf[0]).strip()
        try:
            conf = float(text_conf[1]) * 100.0 if float(text_conf[1]) <= 1.0 else float(text_conf[1])
        except Exception:
            conf = 0.0
        if not text:
            continue
        left = min(point[0] for point in box)
        top = min(point[1] for point in box)
        parsed_lines.append((left, top, text, conf))

    if not parsed_lines:
        return "", 0.0, 0, []

    parsed_lines.sort(key=lambda item: (item[1], item[0]))
    raw_text = " ".join(item[2] for item in parsed_lines)
    confidence_values = [item[3] for item in parsed_lines if item[3] > 0]
    confidence = float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
    return raw_text, confidence, len(parsed_lines), [(item[2], item[3]) for item in parsed_lines]


def _choose_best_row_ocr(row_color: np.ndarray, row_gray: np.ndarray) -> Tuple[str, float, int, str]:
    candidate_images = _build_row_ocr_inputs(row_color, row_gray)
    if not candidate_images:
        return "", 0.0, 0, ""

    best_raw = ""
    best_text = ""
    best_confidence = 0.0
    best_segments = 0

    for image in candidate_images[:2]:
        raw_text, confidence, segment_count, _ = _ocr_with_paddle(image)
        cleaned_text = _normalize_move_like_text(raw_text)
        score = (confidence, float(len(cleaned_text)))
        best_score = (best_confidence, float(len(best_text)))
        if score > best_score:
            best_raw = raw_text
            best_text = cleaned_text
            best_confidence = confidence
            best_segments = segment_count

        if best_confidence >= 65.0 and best_text.strip():
            break

    return best_raw, best_confidence, best_segments, best_text


def _render_visualization(
    table_gray: np.ndarray,
    rows: List[OCRRowResult],
    output_dir: str | Path,
    prefix: str,
) -> str:
    ensure_dir(output_dir)
    canvas = cv2.cvtColor(table_gray, cv2.COLOR_GRAY2BGR)

    for row in rows:
        x1, y1, x2, y2 = row.bbox
        if row.text.strip():
            color = (0, 180, 0) if row.confidence >= 60 else (0, 165, 255) if row.confidence >= 35 else (0, 0, 255)
        else:
            color = (160, 160, 160)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        label = f"r{row.row_index:02d}:{row.confidence:.0f}"
        cv2.putText(canvas, label, (x1 + 2, max(12, y1 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        if row.text.strip():
            snippet = row.text.strip()[:36]
            cv2.putText(canvas, snippet, (x1 + 2, min(canvas.shape[0] - 4, y2 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    output_path = Path(output_dir) / f"{prefix}_row_ocr_visualization.png"
    cv2.imwrite(str(output_path), canvas)
    return str(output_path)


def _assemble_text(rows: List[OCRRowResult]) -> str:
    ordered_rows = sorted(rows, key=lambda item: item.row_index)
    lines = [row.text.strip() for row in ordered_rows if row.text.strip()]
    return "\n".join(lines)


def extract_scoresheet_moves(
    preprocess_result: PreprocessResult,
    debug_dir: str | Path | None = None,
    prefix: str = "img",
) -> OCRTableResult:
    warnings: List[str] = []

    table_gray = preprocess_result.table_gray
    table_color = preprocess_result.table_color
    table_cleaned = preprocess_result.table_cleaned

    table_horizontal = crop_region(preprocess_result.horizontal_lines, preprocess_result.table_bbox, pad=6)
    row_centers = _detect_line_centers(table_horizontal, "horizontal", threshold_ratio=0.30)

    height, width = table_cleaned.shape[:2]
    row_intervals = _build_intervals(row_centers, height, min_span=max(20, height // 70), margin=2)

    if len(row_intervals) < 2:
        warnings.append("Could not detect enough row separators; using equal-height fallback rows.")
        row_count = max(8, height // 48)
        row_intervals = _equal_intervals(height, row_count)

    rows: List[OCRRowResult] = []
    row_debug_dir = None
    if debug_dir is not None:
        row_debug_dir = Path(debug_dir) / "rows"
        ensure_dir(row_debug_dir)

    for row_index, row_interval in enumerate(row_intervals):
        top, bottom = row_interval
        row_gray = table_gray[top:bottom, :].copy()
        row_color = table_color[top:bottom, :].copy() if table_color.size else None
        row_thresholded = table_cleaned[top:bottom, :].copy()

        if row_gray.size == 0:
            continue

        inner_pad = 14
        if row_gray.shape[1] > inner_pad * 2:
            row_gray = row_gray[:, inner_pad : row_gray.shape[1] - inner_pad].copy()
            row_thresholded = row_thresholded[:, inner_pad : row_thresholded.shape[1] - inner_pad].copy()
            if row_color is not None and row_color.size:
                row_color = row_color[:, inner_pad : row_color.shape[1] - inner_pad].copy()

        if row_debug_dir is not None:
            cv2.imwrite(str(row_debug_dir / f"{prefix}_row_{row_index:02d}_crop.png"), row_gray)
            cv2.imwrite(str(row_debug_dir / f"{prefix}_row_{row_index:02d}_thresholded.png"), row_thresholded)
            if row_color is not None and row_color.size:
                cv2.imwrite(str(row_debug_dir / f"{prefix}_row_{row_index:02d}_color.png"), row_color)

        ink_ratio = float(np.mean(row_thresholded < 245)) if row_thresholded.size else 0.0
        if ink_ratio < 0.001:
            rows.append(
                OCRRowResult(
                    row_index=row_index,
                    bbox=(0, top, width, bottom),
                    text="",
                    confidence=0.0,
                    raw_text="",
                    psm=8,
                )
            )
            continue

        raw_text, confidence, segments, cleaned_text = _choose_best_row_ocr(
            row_color if row_color is not None and row_color.size else row_gray,
            row_gray,
        )

        rows.append(
            OCRRowResult(
                row_index=row_index,
                bbox=(0, top, width, bottom),
                text=cleaned_text,
                confidence=confidence,
                raw_text=raw_text,
                psm=8 if segments else 7,
            )
        )

        if row_debug_dir is not None:
            ocr_input_name = f"{prefix}_row_{row_index:02d}_ocr_input.png"
            best_inputs = _build_row_ocr_inputs(
                row_color if row_color is not None and row_color.size else row_gray,
                row_gray,
            )
            if best_inputs:
                cv2.imwrite(str(row_debug_dir / ocr_input_name), best_inputs[0])

    if not rows:
        warnings.append("No OCR rows were extracted from the detected scoresheet table.")

    assembled_text = _assemble_text(rows)
    positive_confidences = [row.confidence for row in rows if row.text.strip() and row.confidence > 0]
    average_confidence = float(sum(positive_confidences) / len(positive_confidences)) if positive_confidences else 0.0

    visualization_path: Optional[str] = None
    if debug_dir is not None:
        visualization_path = _render_visualization(table_gray, rows, debug_dir, prefix)

    return OCRTableResult(
        rows=rows,
        assembled_text=assembled_text,
        average_confidence=average_confidence,
        warnings=warnings,
        visualization_path=visualization_path,
        table_bbox=preprocess_result.table_bbox,
    )
