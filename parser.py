from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import regex as re


MOVE_NUMBER_RE = re.compile(r"^\d+(?:\.\.\.|\.)?$")
RESULT_RE = re.compile(r"^(1-0|0-1|1/2-1/2|\*)$")
TOKEN_RE = re.compile(r"[A-Za-z0-9OoxX#+=\-/]+")


@dataclass
class ParseResult:
    tokens: List[str]
    result_token: Optional[str]
    warnings: List[str]


def _normalize_token(token: str) -> str:
    token = token.strip().strip(".,;:[](){}")
    token = token.replace("×", "x")
    token = token.replace("0-0-0", "O-O-O")
    token = token.replace("0-0", "O-O")
    token = token.replace("o-o-o", "O-O-O")
    token = token.replace("o-o", "O-O")
    token = token.replace("!", "")
    token = token.replace("?", "")
    token = re.sub(r"\s+", "", token)
    return token


def parse_moves_from_text(text: str) -> ParseResult:
    warnings: List[str] = []
    extracted = TOKEN_RE.findall(text)

    if not extracted:
        return ParseResult(tokens=[], result_token=None, warnings=["No candidate move tokens found in OCR text."])

    tokens: List[str] = []
    result_token: Optional[str] = None

    for raw in extracted:
        token = _normalize_token(raw)
        if not token:
            continue

        if MOVE_NUMBER_RE.match(token):
            continue

        if RESULT_RE.match(token):
            result_token = token
            continue

        if token in {"...", "."}:
            continue

        tokens.append(token)

    if not tokens:
        warnings.append("Tokenizer found content, but no move candidates survived filtering.")

    return ParseResult(tokens=tokens, result_token=result_token, warnings=warnings)
