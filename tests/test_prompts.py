"""Tests for prompt templates and phase detection."""

from __future__ import annotations

import chess

from chess_evolve.prompts import (
    ANALYSIS_PROMPTS,
    BLUNDER_CHECK_PROMPTS,
    CRITIQUE_PROMPTS,
    ENDGAME_HINTS,
    MIDDLEGAME_HINTS,
    OPENING_HINTS,
    POSITIONAL_PROMPTS,
    PROMPT_REGISTRY,
    SELECTOR_PROMPTS,
    TACTICAL_PROMPTS,
    _get_prompt,
    _register_prompt,
    detect_phase,
)


class TestDetectPhase:
    def test_opening(self):
        fen = chess.STARTING_FEN
        assert detect_phase(fen) == "opening"

    def test_middlegame(self, midgame_board):
        assert detect_phase(midgame_board.fen()) == "middlegame"

    def test_endgame(self, endgame_board):
        assert detect_phase(endgame_board.fen()) == "endgame"

    def test_boundary_opening_middlegame(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        pieces = sum(1 for c in fen.split()[0] if c.isalpha() and c.lower() != 'k')
        assert pieces > 26
        assert detect_phase(fen) == "opening"


class TestGetPrompt:
    def test_known_tactical_key(self):
        result = _get_prompt("tactical_style", "broad")
        assert result == TACTICAL_PROMPTS["broad"]

    def test_known_positional_key(self):
        result = _get_prompt("positional_style", "classical")
        assert result == POSITIONAL_PROMPTS["classical"]

    def test_known_verify_key(self):
        result = _get_prompt("verify_style", "strict")
        assert result == BLUNDER_CHECK_PROMPTS["strict"]

    def test_unknown_value_returns_value_itself(self):
        result = _get_prompt("tactical_style", "custom_never_seen_prompt_text")
        assert result == "custom_never_seen_prompt_text"

    def test_unknown_knob_name(self):
        result = _get_prompt("nonexistent_knob", "some_value")
        assert result == "some_value"


class TestRegisterPrompt:
    def test_register_and_retrieve(self, tmp_path, monkeypatch):
        monkeypatch.setattr("chess_evolve.config.LIVE_DIR", tmp_path)
        _register_prompt("tactical_style", "test_variant", "A custom tactical prompt")
        assert PROMPT_REGISTRY["tactical_style"]["test_variant"] == "A custom tactical prompt"
        result = _get_prompt("tactical_style", "test_variant")
        assert result == "A custom tactical prompt"
        # Clean up
        del PROMPT_REGISTRY["tactical_style"]["test_variant"]

    def test_register_writes_recording(self, tmp_path, monkeypatch):
        monkeypatch.setattr("chess_evolve.config.LIVE_DIR", tmp_path)
        _register_prompt("test_knob", "test_val", "test prompt text")
        recording = tmp_path / "recording.jsonl"
        assert recording.exists()
        assert "prompt_invented" in recording.read_text()
        # Clean up
        if "test_knob" in PROMPT_REGISTRY:
            del PROMPT_REGISTRY["test_knob"]


class TestPromptDictsAreStrings:
    def test_analysis_prompts(self):
        for k, v in ANALYSIS_PROMPTS.items():
            assert isinstance(v, str), f"ANALYSIS_PROMPTS[{k}] is not a string"

    def test_tactical_prompts(self):
        for k, v in TACTICAL_PROMPTS.items():
            assert isinstance(v, str), f"TACTICAL_PROMPTS[{k}] is not a string"

    def test_positional_prompts(self):
        for k, v in POSITIONAL_PROMPTS.items():
            assert isinstance(v, str), f"POSITIONAL_PROMPTS[{k}] is not a string"

    def test_opening_hints(self):
        for k, v in OPENING_HINTS.items():
            assert isinstance(v, str), f"OPENING_HINTS[{k}] is not a string"

    def test_middlegame_hints(self):
        for k, v in MIDDLEGAME_HINTS.items():
            assert isinstance(v, str), f"MIDDLEGAME_HINTS[{k}] is not a string"

    def test_endgame_hints(self):
        for k, v in ENDGAME_HINTS.items():
            assert isinstance(v, str), f"ENDGAME_HINTS[{k}] is not a string"

    def test_selector_prompts(self):
        for k, v in SELECTOR_PROMPTS.items():
            assert isinstance(v, str), f"SELECTOR_PROMPTS[{k}] is not a string"

    def test_blunder_check_prompts(self):
        for k, v in BLUNDER_CHECK_PROMPTS.items():
            assert isinstance(v, str), f"BLUNDER_CHECK_PROMPTS[{k}] is not a string"

    def test_critique_prompts(self):
        for k, v in CRITIQUE_PROMPTS.items():
            assert isinstance(v, str), f"CRITIQUE_PROMPTS[{k}] is not a string"
