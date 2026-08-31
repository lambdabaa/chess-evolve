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
        # sum cp: game1=50+40+30-200-300=-380, game2=60+50-100-350=-340, game3=70+60+50+40+30=250
        # all above -500, so total = -380 + -340 + 250 = -470
        assert r.composite_score == -470

    def test_composite_score_with_win(self):
        r = EvalResult(wins=1, draws=0, losses=0, total_moves=20)
        r.games = [{"eval_curve": [100, 150, 200]}]
        assert r.composite_score == 450 + 200_000

    def test_composite_score_with_draw(self):
        r = EvalResult(wins=0, draws=1, losses=0, total_moves=40)
        r.games = [{"eval_curve": [0, 10, -10, 5]}]
        assert r.composite_score == 5 + 100_000

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
