"""Tests for EvalResult scoring and game utilities."""

from __future__ import annotations

from chess_evolve.game import EvalResult, format_moves


class TestEvalResult:
    def test_composite_score_all_losses(self):
        r = EvalResult(wins=0, draws=0, losses=3, total_moves=30)
        r.games = [
            {"eval_curve": [50, 40, 30, -200, -300]},
            {"eval_curve": [60, 50, -100, -350]},
            {"eval_curve": [70, 60, 50, 40, 30]},
        ]
        # (cp+500)/1000: game1=.55+.54+.53+.3+.2=2.12
        # game2=.56+.55+.4+.15=1.66, game3=.57+.56+.55+.54+.53=2.75
        expected = 2.12 + 1.66 + 2.75
        assert abs(r.composite_score - expected) < 0.01

    def test_composite_score_with_win(self):
        r = EvalResult(wins=1, draws=0, losses=0, total_moves=20)
        r.games = [{"eval_curve": [100, 150, 200]}]
        # (.6+.65+.7) + 200
        expected = 1.95 + 200
        assert abs(r.composite_score - expected) < 0.01

    def test_composite_score_with_draw(self):
        r = EvalResult(wins=0, draws=1, losses=0, total_moves=40)
        r.games = [{"eval_curve": [0, 10, -10, 5]}]
        # (.5+.51+.49+.505) + 100
        expected = 2.005 + 100
        assert abs(r.composite_score - expected) < 0.01

    def test_blunder_count(self):
        r = EvalResult()
        r.games = [{"eval_curve": [100, -150, -160, 50, -200]}]
        # drop 1: 100 -> -150 = -250 (blunder)
        # drop 2: -150 -> -160 = -10 (not a blunder)
        # drop 3: -160 -> 50 = +210 (improvement)
        # drop 4: 50 -> -200 = -250 (blunder)
        assert r.blunder_count == 2

    def test_avg_eval_empty(self):
        r = EvalResult()
        r.games = [{"eval_curve": []}]
        assert r.avg_eval == 0

    def test_avg_eval(self):
        r = EvalResult()
        r.games = [{"eval_curve": [100, 200, 300]}]
        assert r.avg_eval == 200.0

    def test_win_rate_string(self):
        r = EvalResult(wins=2, draws=1, losses=3)
        assert r.win_rate == "+2=1-3"

    def test_score_property(self):
        r = EvalResult(wins=1, draws=1, losses=0)
        assert r.score == 0.75  # (1 + 0.5) / 2

    def test_to_cycle_record(self):
        r = EvalResult(wins=1, draws=0, losses=0, total_moves=10)
        r.games = [{"eval_curve": [50, 100], "tag": "test", "result": "win"}]
        cr = r.to_cycle_record(gen=5)
        assert cr.cycle_number == 5
        assert cr.mode == "chess"
        assert cr.kept == 1
        assert cr.reverted == 0
        assert len(cr.steps) > 0
        assert len(cr.experiments) == 1


class TestFormatMoves:
    def test_empty(self):
        assert format_moves([]) == ""

    def test_single_move(self):
        assert format_moves(["e2e4"]) == "1.e2e4"

    def test_full_move_pair(self):
        assert format_moves(["e2e4", "e7e5"]) == "1.e2e4 e7e5"

    def test_multiple_moves(self):
        result = format_moves(["e2e4", "e7e5", "g1f3", "b8c6"])
        assert result == "1.e2e4 e7e5 2.g1f3 b8c6"

    def test_odd_number_of_moves(self):
        result = format_moves(["e2e4", "e7e5", "g1f3"])
        assert result == "1.e2e4 e7e5 2.g1f3"
