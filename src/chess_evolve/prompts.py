"""Prompt registry — starts empty, Opus invents prompts at runtime."""

from __future__ import annotations

import json
import time

PROMPT_REGISTRY: dict[str, dict[str, str]] = {}

GENERATOR_PROMPTS = {
    "generic": (
        "You are playing chess. Look at the board position and legal moves. "
        "Pick a move. Output ONLY the UCI move (e.g. e2e4). Nothing else."
    ),
}

PROMPT_TABLES: dict[str, dict[str, str]] = {
    "generator_style": GENERATOR_PROMPTS,
}


def detect_phase(fen: str) -> str:
    """Detect game phase from piece count in FEN."""
    piece_chars = sum(1 for c in fen.split()[0] if c.isalpha() and c.lower() != 'k')
    if piece_chars > 26:
        return "opening"
    elif piece_chars > 10:
        return "middlegame"
    return "endgame"


def _get_prompt(knob_name: str, value: str) -> str:
    """Look up a prompt by knob name and value. Falls back to dynamic registry."""
    table = PROMPT_TABLES.get(knob_name, {})
    if value in table:
        return table[value]
    if knob_name in PROMPT_REGISTRY and value in PROMPT_REGISTRY[knob_name]:
        return PROMPT_REGISTRY[knob_name][value]
    if value and value not in table:
        return value
    return table.get(list(table.keys())[0], "") if table else ""


def _register_prompt(knob_name: str, value: str, prompt_text: str) -> None:
    """Register a new prompt variant created by Opus."""
    from chess_evolve.config import LIVE_DIR
    from chess_evolve.pipeline import KNOB_SPACE

    PROMPT_REGISTRY.setdefault(knob_name, {})[value] = prompt_text
    for _, (name, choices) in enumerate(KNOB_SPACE):
        if name == knob_name and value not in choices:
            choices.append(value)
            break
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_DIR / "recording.jsonl", "a") as f:
        f.write(json.dumps({
            "t": time.monotonic(), "type": "prompt_invented",
            "knob": knob_name, "value": value, "prompt": prompt_text,
        }) + "\n")
