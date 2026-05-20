from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL import ImageSequence


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def _register_heif_if_available(warnings: List[str]) -> None:
    """Enable HEIC/HEIF support for Pillow if pillow-heif is installed."""
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:
        warnings.append(
            "HEIC/HEIF decoding may be unavailable. Install optional package 'pillow-heif' if needed."
        )


def normalize_ocr_text(text: str) -> str:
    """Normalize common OCR artifacts before parsing."""
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\t": " ",
        "\r": "\n",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Remove obvious non-notation noise characters while keeping punctuation used in SAN.
    text = re.sub(r"[^A-Za-z0-9\s\.=+#\-/:xXOo?!]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_images_from_input(input_path: str | Path) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load one or more images from input path.

    Returns:
        images: list of OpenCV BGR images
        labels: list of page labels
        warnings: list of warning strings
    """
    path = Path(input_path)
    warnings: List[str] = []

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    _register_heif_if_available(warnings)

    if is_pdf(path):
        try:
            from pdf2image import convert_from_path
        except Exception:
            raise RuntimeError(
                "PDF input requires pdf2image. Install it via 'pip install pdf2image' and ensure poppler is available."
            )

        pil_pages = convert_from_path(str(path), dpi=300)
        if not pil_pages:
            raise RuntimeError("No pages were rendered from the PDF.")

        images: List[np.ndarray] = []
        labels: List[str] = []
        for idx, pil_img in enumerate(pil_pages, start=1):
            rgb = np.array(pil_img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            images.append(bgr)
            labels.append(f"page_{idx}")
        return images, labels, warnings

    # Preferred path: Pillow handles more formats than OpenCV and can decode multi-frame files.
    try:
        pil_source = Image.open(path)
        images: List[np.ndarray] = []
        labels: List[str] = []

        for idx, frame in enumerate(ImageSequence.Iterator(pil_source), start=1):
            rgb = np.array(frame.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            images.append(bgr)
            labels.append(path.stem if idx == 1 else f"{path.stem}_{idx}")

        if images:
            if len(images) > 1:
                warnings.append(f"Detected multi-frame image input with {len(images)} frames.")
            return images, labels, warnings
    except Exception:
        pass

    # Fallback: OpenCV decode from bytes for file paths with non-ASCII characters.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            warnings.append("Loaded image via OpenCV byte-decoding fallback.")
            return [image], [path.stem], warnings
    except Exception:
        pass

    raise RuntimeError(
        "Could not decode input file. Supported by default: PNG/JPEG/JPG/TIFF/BMP/WebP/GIF/PDF. "
        "For HEIC/HEIF, install optional dependency 'pillow-heif'."
    )


def save_debug_image(image: np.ndarray, output_dir: str | Path, filename: str) -> Path:
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), image)
    return out_path


def write_text_file(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def format_moves_for_console(moves: List[str], per_line: int = 10) -> str:
    if not moves:
        return "(no moves found)"

    chunks: List[str] = []
    for i in range(0, len(moves), per_line):
        subset = moves[i : i + per_line]
        chunks.append(" ".join(subset))
    return "\n".join(chunks)


def default_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    in_name = Path(input_path).stem
    out_dir = ensure_dir(output_dir)
    return out_dir / f"{in_name}.pgn"


def timestamp_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


def environment_hint_for_tesseract() -> str:
    if os.name == "nt":
        return (
            "On Windows, set TESSDATA_PREFIX or ensure tesseract.exe is installed and in PATH. "
            "Typical path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )
    return "Ensure tesseract is installed and available in PATH."
