"""Shared test fixtures for chess-evolve."""

from __future__ import annotations

import chess
import pytest

from chess_evolve.pipeline import PipelineConfig


@pytest.fixture
def default_config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def start_board() -> chess.Board:
    return chess.Board()


@pytest.fixture
def midgame_board() -> chess.Board:
    return chess.Board("r4rk1/ppp2ppp/2n5/3pp3/2B5/2N2N2/PPPP1PPP/R4RK1 b - - 0 10")


@pytest.fixture
def endgame_board() -> chess.Board:
    return chess.Board("8/5pk1/8/8/8/8/5PK1/8 w - - 0 1")
