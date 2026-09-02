"""Tests for prompt templates and phase detection."""

from __future__ import annotations

import chess

from chess_evolve.prompts import (
    GENERATOR_PROMPTS,
    PROMPT_REGISTRY,
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
    def test_known_generator_key(self):
        result = _get_prompt("generator_style", "generic")
        assert result == GENERATOR_PROMPTS["generic"]

    def test_unknown_value_returns_value_itself(self):
        result = _get_prompt("generator_style", "custom_never_seen_prompt_text")
        assert result == "custom_never_seen_prompt_text"

    def test_unknown_knob_name(self):
        result = _get_prompt("nonexistent_knob", "some_value")
        assert result == "some_value"


class TestRegisterPrompt:
    def test_register_and_retrieve(self, tmp_path, monkeypatch):
        monkeypatch.setattr("chess_evolve.config.LIVE_DIR", tmp_path)
        _register_prompt("generator_style", "test_variant", "A custom prompt")
        assert PROMPT_REGISTRY["generator_style"]["test_variant"] == "A custom prompt"
        result = _get_prompt("generator_style", "test_variant")
        assert result == "A custom prompt"
        del PROMPT_REGISTRY["generator_style"]["test_variant"]

    def test_register_writes_recording(self, tmp_path, monkeypatch):
        monkeypatch.setattr("chess_evolve.config.LIVE_DIR", tmp_path)
        _register_prompt("test_knob", "test_val", "test prompt text")
        recording = tmp_path / "recording.jsonl"
        assert recording.exists()
        assert "prompt_invented" in recording.read_text()
        if "test_knob" in PROMPT_REGISTRY:
            del PROMPT_REGISTRY["test_knob"]


class TestPromptDictsAreStrings:
    def test_generator_prompts(self):
        for k, v in GENERATOR_PROMPTS.items():
            assert isinstance(v, str), f"GENERATOR_PROMPTS[{k}] is not a string"
