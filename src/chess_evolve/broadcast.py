"""Live UI broadcasting: write game state and eval results for the web server."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import chess

from chess_evolve.config import LIVE_DIR
from chess_evolve.prompts import detect_phase

if TYPE_CHECKING:
    from chess_evolve.game import EvalResult
    from chess_evolve.pipeline import PipelineConfig


def broadcast_game_state(
    tag: str, board: chess.Board, game_moves: list[str],
    llm_white: bool, elo: int, result: str | None = None,
    move_count: int = 0, gen: int = 0, config: str = "",
    whose_turn: str = "", active_node: str = "",
    node_outputs: dict[str, str] | None = None,
    eval_curve: list[int] | None = None,
    full_config: str = "",
) -> None:
    """Write game state to a JSON file for the live web UI."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = tag.replace(":", "_").replace("(", "").replace(")", "").replace(" ", "_")[:80]
    is_llm_turn = (board.turn == chess.WHITE) == llm_white
    data = {
        "tag": tag,
        "fen": board.fen(),
        "moves": game_moves,
        "llm_white": llm_white,
        "elo": elo,
        "result": result,
        "move_count": move_count,
        "gen": gen,
        "config": config,
        "full_config": full_config,
        "whose_turn": whose_turn or ("llm" if is_llm_turn else "stockfish"),
        "active_node": active_node,
        "node_outputs": node_outputs or {},
        "eval_curve": eval_curve or [],
        "game_phase": detect_phase(board.fen()),
    }
    (LIVE_DIR / f"{safe_name}.json").write_text(json.dumps(data))
    with open(LIVE_DIR / "recording.jsonl", "a") as f:
        f.write(json.dumps({"t": time.monotonic(), "type": "game", "file": safe_name, **data}) + "\n")


def broadcast_eval_result(
    label: str, cfg: PipelineConfig, result: EvalResult,
    gen: int, is_best: bool = False,
) -> None:
    """Append an eval result to the experiment log for the web UI."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "label": label,
        "gen": gen,
        "score": result.composite_score,
        "win_rate": result.win_rate,
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "errors": result.total_errors,
        "pipelines": result.total_pipeline_runs,
        "elo": cfg.opponent_elo,
        "avg_eval": result.avg_eval,
        "blunders": result.blunder_count,
        "moves": result.total_moves,
        "composite": result.composite_score,
        "config": cfg.label,
        "is_best": is_best,
    }
    log_path = LIVE_DIR / "experiment_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    with open(LIVE_DIR / "recording.jsonl", "a") as f:
        f.write(json.dumps({"t": time.monotonic(), "type": "eval", **entry}) + "\n")
