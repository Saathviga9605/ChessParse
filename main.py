from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from ocr_engine import OCRTableResult, extract_scoresheet_moves
from parser import parse_moves_from_text
from pgn_exporter import export_pgn
from preprocess import PreprocessResult, preprocess_image
from utils import (
    default_output_path,
    ensure_dir,
    environment_hint_for_ocr,
    format_moves_for_console,
    load_images_from_input,
    timestamp_iso,
    write_text_file,
)
from validator import validate_and_correct_moves


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert chess scoresheet images/PDFs to validated PGN.",
    )
    parser.add_argument("--input", required=True, help="Path to input image or PDF")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PGN path. Defaults to sample_outputs/<input_name>.pgn",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_outputs",
        help="Directory for generated PGN and logs when --output is not provided.",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug_outputs",
        help="Optional directory for debug artifacts (preprocess images, OCR text logs).",
    )
    parser.add_argument("--event", default="OCR Reconstructed Game", help="PGN Event header")
    parser.add_argument("--site", default="Local", help="PGN Site header")
    parser.add_argument("--white", default="?", help="PGN White header")
    parser.add_argument("--black", default="?", help="PGN Black header")
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input does not exist: {input_path}")
        return 2

    output_path = Path(args.output) if args.output else default_output_path(input_path, args.output_dir)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        ensure_dir(debug_dir)

    try:
        images, labels, load_warnings = load_images_from_input(input_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load input: {exc}")
        if "paddle" in str(exc).lower() or "ocr" in str(exc).lower() or "pdf" in str(exc).lower():
            print(f"[HINT] {environment_hint_for_ocr()}")
        return 2

    for warning in load_warnings:
        print(f"[WARN] {warning}")

    all_raw_text: List[str] = []
    all_clean_text: List[str] = []
    ocr_conf_values: List[float] = []
    all_rows_log: List[str] = []

    for image, label in zip(images, labels):
        preprocess_result: PreprocessResult = preprocess_image(
            image,
            debug_dir=debug_dir,
            prefix=label,
        )
        metrics = preprocess_result.metrics
        print(
            f"[INFO] Preprocessed {label}: table={int(metrics['table_width'])}x{int(metrics['table_height'])}, "
            f"page={int(metrics['resized_width'])}x{int(metrics['resized_height'])}, "
            f"bbox=({int(metrics['table_x'])}, {int(metrics['table_y'])})"
        )

        try:
            ocr_result: OCRTableResult = extract_scoresheet_moves(
                preprocess_result,
                debug_dir=debug_dir,
                prefix=label,
            )
        except Exception as exc:
            print(f"[ERROR] OCR failed on {label}: {exc}")
            print(f"[HINT] {environment_hint_for_ocr()}")
            return 3

        ocr_conf_values.append(ocr_result.average_confidence)
        all_raw_text.append("\n".join(row.raw_text for row in ocr_result.rows if row.raw_text.strip()))
        all_clean_text.append(ocr_result.assembled_text)

        print(f"[INFO] OCR {label}: average_row_confidence={ocr_result.average_confidence:.1f}, rows={len(ocr_result.rows)}")
        for row in ocr_result.rows:
            if row.text.strip() or row.confidence > 0:
                print(f"[ROW] r{row.row_index:02d} conf={row.confidence:5.1f} text={row.text!r}")
                all_rows_log.append(
                    f"{label}\tr{row.row_index}\tconf={row.confidence:.1f}\ttext={row.text}\traw={row.raw_text.strip()}"
                )

        for warn in preprocess_result.warnings:
            print(f"[WARN] {warn}")
        for warn in ocr_result.warnings:
            print(f"[WARN] {warn}")

    merged_text = "\n".join(text for text in all_clean_text if text.strip())
    parse_result = parse_moves_from_text(merged_text)

    for warn in parse_result.warnings:
        print(f"[WARN] {warn}")

    validation = validate_and_correct_moves(parse_result.tokens)

    for corr in validation.corrections:
        print(f"[FIX] {corr}")
    for warn in validation.warnings:
        print(f"[WARN] {warn}")

    headers = {
        "Event": args.event,
        "Site": args.site,
        "White": args.white,
        "Black": args.black,
        "Annotator": "OCR Chess Scoresheet Parser",
    }
    if parse_result.result_token:
        headers["Result"] = parse_result.result_token

    pgn_text, pgn_warnings = export_pgn(validation.san_moves, output_path=output_path, headers=headers)
    for warn in pgn_warnings:
        print(f"[WARN] {warn}")

    print("\n=== Parsed SAN Moves ===")
    print(format_moves_for_console(validation.san_moves, per_line=12))

    print("\n=== Summary ===")
    print(f"Input: {input_path}")
    print(f"Pages processed: {len(images)}")
    avg_ocr_conf = sum(ocr_conf_values) / len(ocr_conf_values) if ocr_conf_values else 0.0
    print(f"Average OCR confidence: {avg_ocr_conf:.2f}")
    print(f"Validation confidence: {validation.confidence:.2f}")
    print(f"Moves accepted: {len(validation.san_moves)}")
    print(f"Moves skipped: {len(validation.failed_tokens)}")
    print(f"PGN saved: {output_path}")

    if debug_dir is not None:
        write_text_file(debug_dir / "ocr_raw_text.txt", "\n\n---\n\n".join(all_raw_text))
        write_text_file(debug_dir / "ocr_clean_text.txt", merged_text)
        write_text_file(debug_dir / "ocr_rows.txt", "\n".join(all_rows_log))
        write_text_file(
            debug_dir / "pipeline_report.txt",
            "\n".join(
                [
                    f"timestamp={timestamp_iso()}",
                    f"input={input_path}",
                    f"pages={len(images)}",
                    f"avg_ocr_conf={avg_ocr_conf:.2f}",
                    f"validation_conf={validation.confidence:.2f}",
                    f"moves_accepted={len(validation.san_moves)}",
                    f"moves_skipped={len(validation.failed_tokens)}",
                ]
            ),
        )
        print(f"Debug artifacts written to: {debug_dir}")

    # Print final PGN to make quick copy/paste possible in terminals.
    print("\n=== PGN ===")
    print(pgn_text)

    return 0


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    code = run_pipeline(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
