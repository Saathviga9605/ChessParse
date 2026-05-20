from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Set, Tuple

import chess


@dataclass
class ValidationResult:
    san_moves: List[str]
    warnings: List[str]
    corrections: List[str]
    failed_tokens: List[str]
    confidence: float


OCR_CHAR_MAP = {
    "s": "5",
    "S": "5",
    "l": "1",
    "I": "1",
    "|": "1",
    "z": "2",
    "Z": "2",
    "o": "0",
    "O": "0",
    "b": "6",
    "g": "9",
}


def _clean_token(token: str) -> str:
    t = token.strip()
    t = t.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    t = t.replace("o-o-o", "O-O-O").replace("o-o", "O-O")
    t = t.replace(" ", "")
    t = t.replace("!", "").replace("?", "")
    t = t.replace("×", "x")
    return t


def _generate_variants(token: str, max_variants: int = 80) -> List[str]:
    token = _clean_token(token)
    variants: Set[str] = {token}

    if token in {"0-0", "O-0", "0-O", "o-o", "O-O"}:
        variants.add("O-O")
    if token in {"0-0-0", "O-0-0", "0-O-O", "o-o-o", "O-O-O"}:
        variants.add("O-O-O")

    # Single-character substitutions for common OCR confusions.
    chars = list(token)
    for i, ch in enumerate(chars):
        repl = OCR_CHAR_MAP.get(ch)
        if repl is not None:
            alt = chars.copy()
            alt[i] = repl
            variants.add("".join(alt))

    # Inverse map for cases where a digit should be a letter in castling patterns.
    inverse_map = {v: k for k, v in OCR_CHAR_MAP.items()}
    for i, ch in enumerate(chars):
        if ch in inverse_map:
            alt = chars.copy()
            alt[i] = inverse_map[ch]
            variants.add("".join(alt))

    # Common piece letter confusion.
    piece_fixups = {"H": "N", "A": "R", "P": "B"}
    for i, ch in enumerate(chars):
        if ch in piece_fixups:
            alt = chars.copy()
            alt[i] = piece_fixups[ch]
            variants.add("".join(alt))

    if len(variants) > max_variants:
        return list(sorted(variants))[:max_variants]
    return list(sorted(variants))


def _similarity(a: str, b: str) -> float:
    a_norm = a.replace("+", "").replace("#", "")
    b_norm = b.replace("+", "").replace("#", "")
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _best_legal_match(token: str, legal_san: List[str]) -> Tuple[Optional[str], float]:
    best = None
    best_score = 0.0
    for san in legal_san:
        score = _similarity(token, san)
        if score > best_score:
            best = san
            best_score = score
    return best, best_score


def _try_parse_candidate(board: chess.Board, candidate: str) -> Optional[str]:
    try:
        move = board.parse_san(candidate)
    except ValueError:
        return None
    return board.san(move)


def validate_and_correct_moves(tokens: List[str]) -> ValidationResult:
    board = chess.Board()
    san_moves: List[str] = []
    warnings: List[str] = []
    corrections: List[str] = []
    failed_tokens: List[str] = []
    confidence_points: List[float] = []

    for idx, raw_token in enumerate(tokens, start=1):
        token = _clean_token(raw_token)

        # 1) Direct parse.
        direct = _try_parse_candidate(board, token)
        if direct is not None:
            move = board.parse_san(direct)
            board.push(move)
            san_moves.append(direct)
            confidence_points.append(1.0)
            continue

        # 2) Variant parse.
        parsed_variant: Optional[str] = None
        for variant in _generate_variants(token):
            parsed_variant = _try_parse_candidate(board, variant)
            if parsed_variant is not None:
                break

        if parsed_variant is not None:
            move = board.parse_san(parsed_variant)
            board.push(move)
            san_moves.append(parsed_variant)
            corrections.append(f"Move {idx}: '{raw_token}' -> '{parsed_variant}' (variant correction)")
            warnings.append(f"Corrected OCR move at ply {idx}: {raw_token} -> {parsed_variant}")
            confidence_points.append(0.8)
            continue

        # 3) Fuzzy match against legal SAN options.
        legal_san = [board.san(m) for m in board.legal_moves]
        best, score = _best_legal_match(token, legal_san)
        if best is not None and score >= 0.68:
            try:
                move = board.parse_san(best)
                board.push(move)
                san_moves.append(best)
                corrections.append(
                    f"Move {idx}: '{raw_token}' -> '{best}' (fuzzy legality match, score={score:.2f})"
                )
                warnings.append(
                    f"Fuzzy-corrected move at ply {idx}: {raw_token} -> {best} (score={score:.2f})"
                )
                confidence_points.append(max(0.5, min(score, 0.79)))
                continue
            except ValueError:
                pass

        failed_tokens.append(raw_token)
        warnings.append(f"Could not validate token at ply {idx}: '{raw_token}'. Skipping.")
        confidence_points.append(0.0)

    overall_conf = float(sum(confidence_points) / len(confidence_points)) if confidence_points else 0.0

    return ValidationResult(
        san_moves=san_moves,
        warnings=warnings,
        corrections=corrections,
        failed_tokens=failed_tokens,
        confidence=overall_conf,
    )
