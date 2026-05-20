from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from preprocess import PreprocessResult, crop_region
from utils import ensure_dir, normalize_ocr_text


WHITELIST = "abcdefghKQRBNxO-+=#0123456789"


@dataclass
class OCRCellResult:
    row_index: int
    col_index: int
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    raw_text: str
    psm: int


@dataclass
class OCRTableResult:
    cells: List[OCRCellResult]
    assembled_text: str
    average_confidence: float
    warnings: List[str]
    visualization_path: Optional[str]
    table_bbox: Tuple[int, int, int, int]


def _build_tesseract_config(psm: int) -> str:
    return f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST} -c preserve_interword_spaces=1"


def _compute_confidence(data: Dict[str, List[str]]) -> float:
    conf_values: List[float] = []
    for value_text in data.get("conf", []):
        try:
            value = float(value_text)
        except Exception:
            continue
        if value >= 0:
            conf_values.append(value)
    if not conf_values:
        return 0.0
    return float(sum(conf_values) / len(conf_values))


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _blue_ink_enhancement(color_image: np.ndarray) -> np.ndarray:
    if color_image.size == 0 or len(color_image.shape) != 3:
        return np.array([], dtype=np.uint8)

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (80, 18, 25), (145, 255, 255))

    blue, green, red = cv2.split(color_image)
    dominance = cv2.subtract(blue, cv2.addWeighted(green, 0.5, red, 0.5, 0.0))
    dominance = cv2.GaussianBlur(dominance, (3, 3), 0)
    dominance = cv2.normalize(dominance, None, 0, 255, cv2.NORM_MINMAX)

    masked = cv2.bitwise_and(dominance, dominance, mask=blue_mask)
    masked = cv2.equalizeHist(masked)
    return masked.astype(np.uint8)


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


def _crop_cell(image: np.ndarray, row_interval: Tuple[int, int], col_interval: Tuple[int, int]) -> np.ndarray:
    y1, y2 = row_interval
    x1, x2 = col_interval
    return image[y1:y2, x1:x2].copy()


def _trim_to_ink(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink_mask = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(ink_mask)
    if coords is None:
        _, ink_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(ink_mask)
    if coords is None:
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    pad = 4
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(gray.shape[1], x + w + pad)
    bottom = min(gray.shape[0], y + h + pad)
    return gray[top:bottom, left:right].copy()


def _resize_and_pad(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray

    if gray.shape[1] < 180:
        scale = max(2.5, 240.0 / float(max(1, gray.shape[1])))
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

    return cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _prepare_cell_variants(gray_cell: np.ndarray, color_cell: np.ndarray | None = None) -> List[np.ndarray]:
    base_images: List[np.ndarray] = []

    gray = _ensure_gray(gray_cell)
    if gray.size > 0:
        base_images.append(gray)

    if color_cell is not None and color_cell.size > 0:
        blue = _blue_ink_enhancement(color_cell)
        if blue.size > 0:
            base_images.append(blue)

    if not base_images:
        return []

    variants: List[np.ndarray] = []
    for base in base_images:
        current = base.copy()
        if float(np.mean(current)) < 127.0:
            current = cv2.bitwise_not(current)
        current = _trim_to_ink(current)
        if current.size == 0:
            continue

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        current = clahe.apply(current)
        current = cv2.fastNlMeansDenoising(current, None, h=6, templateWindowSize=7, searchWindowSize=21)
        current = _resize_and_pad(current)

        if current.size == 0:
            continue

        adaptive = cv2.adaptiveThreshold(
            current,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        _, otsu = cv2.threshold(current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.extend([current, otsu, adaptive])

    cleaned_variants: List[np.ndarray] = []
    for variant in variants:
        if variant.size == 0:
            continue
        if float(np.mean(variant)) < 127.0:
            variant = cv2.bitwise_not(variant)
        cleaned_variants.append(variant)
    return cleaned_variants


def _ocr_image(image: np.ndarray, psm: int, timeout: float = 0.8) -> Tuple[str, float, int, str]:
    config = _build_tesseract_config(psm)
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT, timeout=timeout)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract executable not found. Install Tesseract and add it to PATH.") from exc
    except RuntimeError:
        return "", 0.0, 0, ""
    except Exception as exc:
        return "", 0.0, 0, ""

    raw_parts = [str(text).strip() for text in data.get("text", []) if str(text).strip()]
    raw_text = " ".join(raw_parts)
    confidence = _compute_confidence(data)
    cleaned_text = normalize_ocr_text(raw_text)
    words_detected = len([text for text in data.get("text", []) if str(text).strip()])
    return raw_text, confidence, words_detected, cleaned_text


def _choose_best_cell_ocr(binary_cell: np.ndarray, gray_cell: np.ndarray, psm_candidates: Sequence[int]) -> Tuple[str, float, int, str, int]:
    best_raw = ""
    best_cleaned = ""
    best_confidence = 0.0
    best_words = 0
    best_psm = int(psm_candidates[0]) if psm_candidates else 7

    candidate_images: List[np.ndarray] = []
    if binary_cell.size:
        candidate_images.extend(_prepare_cell_variants(binary_cell))
    if gray_cell.size:
        candidate_images.extend(_prepare_cell_variants(gray_cell))

    if not candidate_images:
        return "", 0.0, 0, "", best_psm

    attempts = 0
    for image in candidate_images:
        for psm in psm_candidates:
            attempts += 1
            if attempts > 8:
                break
            if image.size == 0:
                continue
            raw_text, confidence, words_detected, cleaned_text = _ocr_image(image, int(psm))
            score = (confidence, float(len(cleaned_text.strip())))
            best_score = (best_confidence, float(len(best_cleaned.strip())))
            if score > best_score:
                best_raw = raw_text
                best_cleaned = cleaned_text
                best_confidence = confidence
                best_words = words_detected
                best_psm = int(psm)
            if best_confidence >= 55.0 and best_cleaned.strip():
                return best_raw, best_confidence, best_words, best_cleaned, best_psm

        if attempts > 8:
            break

    return best_raw, best_confidence, best_words, best_cleaned, best_psm


def _render_visualization(
    table_gray: np.ndarray,
    cells: List[OCRCellResult],
    output_dir: str | Path,
    prefix: str,
) -> str:
    ensure_dir(output_dir)
    canvas = cv2.cvtColor(table_gray, cv2.COLOR_GRAY2BGR)

    for cell in cells:
        x1, y1, x2, y2 = cell.bbox
        if cell.text.strip():
            color = (0, 180, 0) if cell.confidence >= 60 else (0, 165, 255) if cell.confidence >= 35 else (0, 0, 255)
        else:
            color = (160, 160, 160)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        label = f"{cell.row_index},{cell.col_index}:{cell.confidence:.0f}"
        cv2.putText(canvas, label, (x1 + 2, max(12, y1 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        if cell.text.strip():
            snippet = cell.text.strip()[:14]
            cv2.putText(canvas, snippet, (x1 + 2, min(canvas.shape[0] - 4, y2 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    output_path = Path(output_dir) / f"{prefix}_ocr_visualization.png"
    cv2.imwrite(str(output_path), canvas)
    return str(output_path)


def _assemble_text(cells: List[OCRCellResult]) -> str:
    rows: Dict[int, List[OCRCellResult]] = defaultdict(list)
    for cell in cells:
        rows[cell.row_index].append(cell)

    lines: List[str] = []
    for row_index in sorted(rows):
        ordered = sorted(rows[row_index], key=lambda item: item.col_index)
        row_text = " ".join(cell.text for cell in ordered if cell.text.strip())
        if row_text.strip():
            lines.append(row_text.strip())
    return "\n".join(lines)


def extract_scoresheet_moves(
    preprocess_result: PreprocessResult,
    psm_candidates: Sequence[int] | None = None,
    debug_dir: str | Path | None = None,
    prefix: str = "img",
) -> OCRTableResult:
    warnings: List[str] = []
    psm_list = list(dict.fromkeys(int(psm) for psm in (psm_candidates or (7, 8, 13, 6))))
    if not psm_list:
        psm_list = [7, 8, 13, 6]

    table_gray = preprocess_result.table_gray
    table_color = preprocess_result.table_color
    table_cleaned = preprocess_result.table_cleaned
    table_horizontal = crop_region(preprocess_result.horizontal_lines, preprocess_result.table_bbox, pad=6)
    table_vertical = crop_region(preprocess_result.vertical_lines, preprocess_result.table_bbox, pad=6)

    row_centers = _detect_line_centers(table_horizontal, "horizontal", threshold_ratio=0.30)
    col_centers = _detect_line_centers(table_vertical, "vertical", threshold_ratio=0.18)

    height, width = table_cleaned.shape[:2]
    row_intervals = _build_intervals(row_centers, height, min_span=max(18, height // 90), margin=2)
    col_intervals = _build_intervals(col_centers, width, min_span=max(24, width // 10), margin=2)

    if len(row_intervals) < 2:
        warnings.append("Could not detect enough row separators; using equal-height fallback rows.")
        row_count = max(8, height // 52)
        row_intervals = _equal_intervals(height, row_count)

    if len(col_intervals) < 2:
        warnings.append("Could not detect enough column separators; using three fallback columns.")
        col_intervals = _equal_intervals(width, 3)

    cells: List[OCRCellResult] = []
    for row_index, row_interval in enumerate(row_intervals):
        for col_index, col_interval in enumerate(col_intervals):
            if len(col_intervals) >= 3 and col_index == 0:
                continue
            y1, y2 = row_interval
            x1, x2 = col_interval
            binary_cell = _crop_cell(table_cleaned, row_interval, col_interval)
            gray_cell = _crop_cell(table_gray, row_interval, col_interval)
            color_cell = _crop_cell(table_color, row_interval, col_interval) if table_color.size else None

            if binary_cell.size == 0 and gray_cell.size == 0 and (color_cell is None or color_cell.size == 0):
                continue

            ink_ratio = float(np.mean(binary_cell < 245))
            if ink_ratio < 0.002:
                cells.append(
                    OCRCellResult(
                        row_index=row_index,
                        col_index=col_index,
                        bbox=(x1, y1, x2, y2),
                        text="",
                        confidence=0.0,
                        raw_text="",
                        psm=psm_list[0],
                    )
                )
                continue

            raw_text, confidence, _, cleaned_text, best_psm = _choose_best_cell_ocr(
                binary_cell if binary_cell.size else gray_cell,
                gray_cell,
                psm_list,
            )

            if color_cell is not None and color_cell.size:
                color_raw, color_confidence, _, color_cleaned, color_psm = _choose_best_cell_ocr(
                    color_cell,
                    gray_cell,
                    psm_list,
                )
                if (color_confidence, len(color_cleaned.strip())) > (confidence, len(cleaned_text.strip())):
                    raw_text, confidence, cleaned_text, best_psm = color_raw, color_confidence, color_cleaned, color_psm
            cells.append(
                OCRCellResult(
                    row_index=row_index,
                    col_index=col_index,
                    bbox=(x1, y1, x2, y2),
                    text=cleaned_text,
                    confidence=confidence,
                    raw_text=raw_text,
                    psm=best_psm,
                )
            )

    if not cells:
        warnings.append("No OCR cells were extracted from the detected scoresheet table.")

    assembled_text = _assemble_text(cells)
    positive_confidences = [cell.confidence for cell in cells if cell.text.strip() and cell.confidence > 0]
    if positive_confidences:
        average_confidence = float(sum(positive_confidences) / len(positive_confidences))
    else:
        average_confidence = 0.0

    visualization_path: Optional[str] = None
    if debug_dir is not None:
        visualization_path = _render_visualization(table_gray, cells, debug_dir, prefix)

    return OCRTableResult(
        cells=cells,
        assembled_text=assembled_text,
        average_confidence=average_confidence,
        warnings=warnings,
        visualization_path=visualization_path,
        table_bbox=preprocess_result.table_bbox,
    )
