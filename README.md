# Chess Scoresheet OCR to PGN (Local Python Project)

This project converts handwritten or printed chess scoresheet images (and optionally PDFs) into validated PGN files.

It is fully local and uses only practical Python computer vision/OCR libraries:
- OpenCV
- pytesseract
- python-chess
- Pillow
- regex
- numpy
- optional pdf2image for PDF inputs
- optional pillow-heif for HEIC/HEIF inputs

## Features

- End-to-end local OCR pipeline from image/PDF to PGN
- Broad input format support (png, jpg, jpeg, tiff, bmp, webp, gif, heic/heif, pdf)
- Image preprocessing for OCR robustness:
  - grayscale
  - contrast enhancement (CLAHE)
  - denoising
  - adaptive thresholding
  - deskew
  - upscaling
- OCR extraction with chess-notation-focused Tesseract config
- Regex-driven move token parsing
- OCR error correction heuristics
- Move-by-move legality validation using `python-chess`
- PGN export with metadata headers
- CLI workflow with warnings and debug artifacts
- Optional PDF support via `pdf2image`

## Project Structure

```
project/
|
|-- main.py
|-- preprocess.py
|-- ocr_engine.py
|-- parser.py
|-- validator.py
|-- pgn_exporter.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- sample_outputs/
`-- input_samples/
```

## Requirements

- Python 3.11+
- Tesseract OCR installed on your system

## Setup (Windows)

### 1) Create and activate a virtual environment

```powershell
cd d:\CHESS\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install Tesseract OCR

1. Download and install Tesseract for Windows (commonly from UB Mannheim builds).
2. Typical install path:
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Add Tesseract folder to PATH (System Environment Variables), or set path at runtime.
4. Verify:

```powershell
tesseract --version
```

If `tesseract` is not found, restart terminal after updating PATH.

## Optional: PDF Support Notes

PDF support uses `pdf2image`, which requires Poppler tools available on PATH.

### Install Poppler on Windows

1. Download a Windows Poppler build.
2. Extract it.
3. Add the `bin` folder to PATH.
4. Verify:

```powershell
pdfinfo -v
```

If PDF conversion fails, check Poppler installation first.

## Optional: HEIC/HEIF Support Notes

Some systems cannot decode HEIC/HEIF through Pillow by default. Install:

```powershell
pip install pillow-heif
```

The loader auto-registers HEIC support when this package is available.

## Usage

### Basic image input

```powershell
python main.py --input input_samples\sample.jpg
```

### PDF input

```powershell
python main.py --input input_samples\scoresheet.pdf
```

### Custom output file

```powershell
python main.py --input input_samples\sample.jpg --output sample_outputs\game1.pgn
```

### Enable debug output

```powershell
python main.py --input input_samples\sample.jpg --debug-dir sample_outputs\debug_run_01
```

### Add PGN metadata

```powershell
python main.py --input input_samples\sample.jpg --white "Player A" --black "Player B" --event "Club Match"
```

## CLI Arguments

- `--input` (required): image or PDF path
- `--output`: explicit PGN output path
- `--output-dir`: output directory when `--output` is omitted (default: `sample_outputs`)
- `--debug-dir`: save preprocessing images and OCR logs
- `--psm`: Tesseract page segmentation mode (default: 6)
- `--event`, `--site`, `--white`, `--black`: PGN header fields

## Pipeline Overview

1. Load input image(s):
   - single image, or multiple PDF pages
2. Preprocess each page:
   - grayscale -> CLAHE -> denoise -> adaptive threshold -> deskew -> upscale
3. OCR extraction:
   - Tesseract with a chess-character whitelist
   - confidence estimation from OCR data
4. Parse move tokens:
   - regex extraction of move numbers, SAN-like moves, and game result tokens
5. Validate and correct:
   - direct SAN parse on current board
   - OCR variant substitution attempts
   - fuzzy best legal SAN recovery
6. Export PGN:
   - reconstruct game with `python-chess`
   - write `.pgn` file

## OCR Error Handling and Recovery

The validator attempts to recover common OCR mistakes, including:
- `0-0` vs `O-O`
- `0-0-0` vs `O-O-O`
- rank confusion like `s -> 5`, `l/I -> 1`, `z -> 2`
- fuzzy match against legal SAN moves in current board position

Unrecoverable tokens are skipped with warnings, and processing continues.

## Debug Artifacts

When `--debug-dir` is provided, the pipeline writes:
- preprocessing snapshots per page
- `ocr_raw_text.txt`
- `ocr_clean_text.txt`
- `pipeline_report.txt`

This makes troubleshooting OCR quality and parser behavior easier.

## Example Output Summary

Typical terminal output includes:
- preprocessing metrics (estimated skew, resize)
- OCR confidence per page
- correction logs (`[FIX] ...`)
- warnings for skipped/unreadable moves
- accepted SAN move list
- final PGN text
- output file path

## Limitations

- Extremely poor handwriting can still defeat OCR.
- If move columns are severely misaligned, token order may degrade.
- SAN-only approach may fail when notation is highly abbreviated or inconsistent.
- Fuzzy correction is conservative; some legal but wrong moves are still possible in noisy cases.

## Future Improvements

- Better line/column segmentation to keep white/black move pairing
- Ensemble preprocessing profiles and automatic best-profile selection
- Optional board-state scoring to penalize implausible tactical jumps
- Better move confidence model at token level
- Unit tests for parser and validator edge cases

## Troubleshooting

### `Tesseract executable not found`

- Ensure Tesseract is installed
- Add install directory to PATH
- Restart terminal

### Empty or very low-quality OCR output

- Increase scan quality (300+ DPI)
- Try cleaner crop and stronger contrast
- Use `--debug-dir` and inspect thresholded images

### PDF conversion errors

- Install Poppler and ensure its `bin` path is available
- Verify with `pdfinfo -v`

## License

For local/internal use. Add your organization's license as needed.
