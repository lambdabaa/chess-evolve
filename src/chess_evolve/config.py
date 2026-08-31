"""Constants and configuration for chess-evolve."""

from __future__ import annotations

import os
from pathlib import Path

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
ELO_OPTIONS = [1320, 1420, 1520, 1620]
GAMES_PER_EVAL = 1
MAX_MOVES = 60
NUM_GENERATIONS = 6
CANDIDATES_PER_GEN = 5
WORKSPACE = Path(os.environ.get("CHESS_WORKSPACE", "/tmp/chess-factory"))
LIVE_DIR = WORKSPACE / "live"
