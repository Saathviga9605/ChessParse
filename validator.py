from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Set, Tuple

import chess
import regex as re


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

PIECE_LETTERS = {"K", "Q", "R", "B", "N"}


def _clean_token(token: str) -> str:
    cleaned = token.strip().strip(".,;:[](){}")
    cleaned = cleaned.replace("×", "x")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("!", "").replace("?", "")
    cleaned = cleaned.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    cleaned = cleaned.replace("o-o-o", "O-O-O").replace("o-o", "O-O")
    return cleaned


def _generate_variants(token: str, max_variants: int = 120) -> List[str]:
    token = _clean_token(token)
    variants: Set[str] = {token}

    if token in {"0-0", "O-0", "0-O", "o-o", "O-O"}:
        variants.add("O-O")
    if token in {"0-0-0", "O-0-0", "0-O-O", "o-o-o", "O-O-O"}:
        variants.add("O-O-O")

    if token and token[0].lower() in {"k", "q", "r", "b", "n"}:
        variants.add(token[0].upper() + token[1:])

    promotion_match = re.match(r"^([a-h]x?[a-h]?[18])([qrbnQRBN])$", token)
    if promotion_match:
        variants.add(f"{promotion_match.group(1)}={promotion_match.group(2).upper()}")

    chars = list(token)
    for index, char in enumerate(chars):
        replacement = OCR_CHAR_MAP.get(char)
        if replacement is not None:
            candidate = chars.copy()
            candidate[index] = replacement
            variants.add("".join(candidate))

    inverse_map = {value: key for key, value in OCR_CHAR_MAP.items()}
    for index, char in enumerate(chars):
        replacement = inverse_map.get(char)
        if replacement is not None:
            candidate = chars.copy()
            candidate[index] = replacement
            variants.add("".join(candidate))

    if len(token) >= 2 and token[0] in {"h", "k", "q", "r", "b", "n"}:
        variants.add(token[0].upper() + token[1:])

    if len(variants) > max_variants:
        return sorted(variants)[:max_variants]
    return sorted(variants)


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

    for index, raw_token in enumerate(tokens, start=1):
        token = _clean_token(raw_token)
        if not token:
            continue

        direct = _try_parse_candidate(board, token)
        if direct is not None:
            move = board.parse_san(direct)
            board.push(move)
            san_moves.append(direct)
            confidence_points.append(1.0)
            continue

        corrected: Optional[str] = None
        for variant in _generate_variants(token):
            corrected = _try_parse_candidate(board, variant)
            if corrected is not None:
                if variant != token:
                    corrections.append(f"Move {index}: '{raw_token}' -> '{variant}'")
                    warnings.append(f"Corrected OCR move at ply {index}: {raw_token} -> {variant}")
                break

        if corrected is not None:
            move = board.parse_san(corrected)
            board.push(move)
            san_moves.append(corrected)
            confidence_points.append(0.82)
            continue

        legal_san = [board.san(move) for move in board.legal_moves]
        best_match, score = _best_legal_match(token, legal_san)
        if best_match is not None and score >= 0.68:
            try:
                move = board.parse_san(best_match)
                board.push(move)
                san_moves.append(best_match)
                corrections.append(
                    f"Move {index}: '{raw_token}' -> '{best_match}' (fuzzy legality match, score={score:.2f})"
                )
                warnings.append(
                    f"Fuzzy-corrected move at ply {index}: {raw_token} -> {best_match} (score={score:.2f})"
                )
                confidence_points.append(max(0.5, min(score, 0.79)))
                continue
            except ValueError:
                pass

        failed_tokens.append(raw_token)
        warnings.append(f"Could not validate token at ply {index}: '{raw_token}'. Skipping.")
        confidence_points.append(0.0)

    overall_confidence = float(sum(confidence_points) / len(confidence_points)) if confidence_points else 0.0

    return ValidationResult(
        san_moves=san_moves,
        warnings=warnings,
        corrections=corrections,
        failed_tokens=failed_tokens,
        confidence=overall_confidence,
    )
