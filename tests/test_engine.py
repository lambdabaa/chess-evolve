"""Tests for move extraction and board rendering."""

from __future__ import annotations

import chess

from chess_evolve.engine import _ascii_board, _board_user_msg, _extract_move
from chess_evolve.pipeline import PipelineConfig


class TestExtractMove:
    def test_uci_notation(self, start_board):
        result = _extract_move("I suggest e2e4", start_board)
        assert result == "e2e4"

    def test_uci_in_sentence(self, start_board):
        result = _extract_move("The best move is d2d4 because it controls the center", start_board)
        assert result == "d2d4"

    def test_san_notation(self, start_board):
        result = _extract_move("Nf3", start_board)
        assert result == "g1f3"

    def test_san_pawn(self, start_board):
        result = _extract_move("e4", start_board)
        assert result == "e2e4"

    def test_gibberish_returns_none(self, start_board):
        result = _extract_move("I'm not sure what to play here", start_board)
        assert result is None

    def test_illegal_move_returns_none(self, start_board):
        result = _extract_move("e5e6", start_board)
        assert result is None

    def test_multiple_moves_returns_first_legal(self, start_board):
        result = _extract_move("Options: e2e4 d2d4 g1f3", start_board)
        assert result == "e2e4"

    def test_promotion_move(self):
        board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        result = _extract_move("a7a8q", board)
        assert result == "a7a8q"

    def test_midgame_position(self, midgame_board):
        result = _extract_move("d5d4", midgame_board)
        assert result == "d5d4"


class TestAsciiBoard:
    def test_returns_string(self, start_board):
        result = _ascii_board(start_board)
        assert isinstance(result, str)

    def test_has_coordinates(self, start_board):
        result = _ascii_board(start_board)
        assert "a b c d e f g h" in result

    def test_has_rank_numbers(self, start_board):
        result = _ascii_board(start_board)
        for rank in range(1, 9):
            assert str(rank) in result

    def test_line_count(self, start_board):
        result = _ascii_board(start_board)
        lines = result.strip().split("\n")
        assert len(lines) == 10  # 8 ranks + 2 coordinate lines


class TestBoardUserMsg:
    def test_includes_fen(self, start_board):
        msg = _board_user_msg(start_board)
        assert "rnbqkbnr" in msg

    def test_includes_legal_moves(self, start_board):
        msg = _board_user_msg(start_board)
        assert "e2e4" in msg

    def test_includes_phase_hint(self, start_board):
        cfg = PipelineConfig()
        msg = _board_user_msg(start_board, cfg=cfg)
        assert "OPENING" in msg

    def test_includes_side_to_move(self, start_board):
        msg = _board_user_msg(start_board)
        assert "White" in msg

    def test_endgame_hint(self, endgame_board):
        msg = _board_user_msg(endgame_board)
        assert "ENDGAME" in msg
