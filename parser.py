from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import regex as re


MOVE_NUMBER_RE = re.compile(r"^\d+\.(?:\.\.)?$")
RESULT_RE = re.compile(r"^(1-0|0-1|1/2-1/2|\*)$")

SAN_TOKEN_RE = re.compile(
    r"""
    (?ix)
    ^(
        O-O-O|O-O|0-0-0|0-0|
        [KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|
        [a-h](?:x[a-h])?[1-8](?:=[QRBN])?[+#]?|
        [KQRBN][a-h][1-8][+#]?|
        [KQRBN]x[a-h][1-8][+#]?
    )$
    """
)

TOKENIZER_RE = re.compile(
    r"""
    (?ix)
    (O-O-O|O-O|0-0-0|0-0|1/2-1/2|1-0|0-1|\*|\d+\.\.\.|\d+\.|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)
    """
)


@dataclass
class ParseResult:
    tokens: List[str]
    result_token: Optional[str]
    warnings: List[str]


def _strip_annotations(token: str) -> str:
    return token.strip().replace("!", "").replace("?", "")


def _normalize_token(token: str) -> str:
    t = token.strip()
    t = t.replace("0-0-0", "O-O-O")
    t = t.replace("0-0", "O-O")
    t = t.replace("o-o-o", "O-O-O")
    t = t.replace("o-o", "O-O")
    t = _strip_annotations(t)
    return t


def parse_moves_from_text(text: str) -> ParseResult:
    warnings: List[str] = []
    extracted = TOKENIZER_RE.findall(text)

    if not extracted:
        return ParseResult(tokens=[], result_token=None, warnings=["No candidate move tokens found in OCR text."])

    moves: List[str] = []
    result_token: Optional[str] = None

    for raw in extracted:
        tok = _normalize_token(raw)
        if not tok:
            continue

        if MOVE_NUMBER_RE.match(tok):
            continue

        if RESULT_RE.match(tok):
            result_token = tok
            continue

        if SAN_TOKEN_RE.match(tok):
            moves.append(tok)
        else:
            warnings.append(f"Dropped unrecognized token: {tok}")

    if not moves:
        warnings.append("Tokenizer found content, but no SAN-like moves survived filtering.")

    return ParseResult(tokens=moves, result_token=result_token, warnings=warnings)
