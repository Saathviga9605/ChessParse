from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import chess.pgn


def build_game_from_san_moves(san_moves: List[str], headers: Dict[str, str] | None = None) -> Tuple[chess.pgn.Game, List[str]]:
    game = chess.pgn.Game()
    warnings: List[str] = []

    default_headers = {
        "Event": "OCR Reconstructed Game",
        "Site": "Local",
        "Date": datetime.now().strftime("%Y.%m.%d"),
        "Round": "?",
        "White": "?",
        "Black": "?",
        "Result": "*",
    }

    final_headers = dict(default_headers)
    if headers:
        final_headers.update(headers)

    for key, value in final_headers.items():
        game.headers[key] = value

    board = chess.Board()
    node = game

    for idx, san in enumerate(san_moves, start=1):
        try:
            move = board.parse_san(san)
            board.push(move)
            node = node.add_variation(move)
        except ValueError:
            warnings.append(f"Failed to add SAN move {idx}: {san}")
            break

    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
        game.headers["Result"] = result
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        game.headers["Result"] = "1/2-1/2"

    return game, warnings


def game_to_pgn_string(game: chess.pgn.Game) -> str:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


def export_pgn(
    san_moves: List[str],
    output_path: str | Path,
    headers: Dict[str, str] | None = None,
) -> Tuple[str, List[str]]:
    game, warnings = build_game_from_san_moves(san_moves, headers=headers)
    pgn_text = game_to_pgn_string(game)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pgn_text + "\n", encoding="utf-8")

    return pgn_text, warnings
