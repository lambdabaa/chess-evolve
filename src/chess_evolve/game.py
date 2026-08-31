"""Game playing and evaluation: EvalResult, play_game(), evaluate_pipeline()."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import chess
import chess.engine
from factory.cycle_analyzer import AgentStep, CycleRecord, ExperimentRecord
from factory.workflow.package import Package

from chess_evolve.broadcast import broadcast_game_state
from chess_evolve.config import GAMES_PER_EVAL, MAX_MOVES, STOCKFISH_PATH, WORKSPACE
from chess_evolve.display import DIM, RESET, WHITE, print
from chess_evolve.engine import (
    _board_user_msg,
    _call_llm,
    get_pipeline_move,
    setup_workspace,
)
from chess_evolve.pipeline import PipelineConfig
from chess_evolve.prompts import GAME_PLAN_PROMPT, OPPONENT_MODEL_PROMPT

PIECE_MAP = {
    "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔", "P": "♙",
    "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚", "p": "♟",
}


def format_moves(game_moves: list[str]) -> str:
    """Format a move list as standard chess notation: 1.e2e4 e7e5 2.g1f3 ..."""
    paired = []
    for m in range(0, len(game_moves), 2):
        num = m // 2 + 1
        white = game_moves[m]
        black = game_moves[m + 1] if m + 1 < len(game_moves) else ""
        paired.append(f"{num}.{white} {black}".strip())
    return " ".join(paired)


def render_board(board: chess.Board, llm_white: bool, elo: int, tag: str = "") -> str:
    """Render the board as a unicode diagram with player labels."""
    top_label = f"Stockfish {elo}" if llm_white else "Haiku (LLM)"
    bot_label = "Haiku (LLM)" if llm_white else f"Stockfish {elo}"
    prefix = f"[{tag}] " if tag else ""
    lines = []
    lines.append(f"      {prefix}{top_label}")
    lines.append("      ┌───┬───┬───┬───┬───┬───┬───┬───┐")
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            sym = PIECE_MAP.get(piece.symbol(), piece.symbol()) if piece else " "
            cells.append(f" {sym} ")
        lines.append(f"    {rank+1} │" + "│".join(cells) + "│")
        if rank > 0:
            lines.append("      ├───┼───┼───┼───┼───┼───┼───┼───┤")
    lines.append("      └───┴───┴───┴───┴───┴───┴───┴───┘")
    lines.append("        a   b   c   d   e   f   g   h")
    lines.append(f"      {prefix}{bot_label}")
    return "\n".join(lines)


@dataclass
class EvalResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_moves: int = 0
    total_errors: int = 0
    total_pipeline_runs: int = 0
    games: list[dict] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = self.wins + self.draws + self.losses
        return (self.wins + 0.5 * self.draws) / max(total, 1)

    @property
    def avg_eval(self) -> float:
        all_evals = []
        for g in self.games:
            all_evals.extend(g.get("eval_curve", []))
        return sum(all_evals) / max(len(all_evals), 1)

    @property
    def blunder_count(self) -> int:
        total = 0
        for g in self.games:
            curve = g.get("eval_curve", [])
            total += sum(1 for i in range(1, len(curve)) if curve[i] - curve[i-1] < -200)
        return total

    @property
    def composite_score(self) -> float:
        base = self.avg_eval - 20 * self.blunder_count + 8 * self.total_moves
        return base + 500 * self.wins + 200 * self.draws

    def to_cycle_record(self, gen: int = 0) -> CycleRecord:
        """Build a factory CycleRecord from chess results.

        Includes move history, eval curve, and blunder context so the
        reflector can see exactly where and why games went wrong.
        """
        steps = []
        experiments = []
        order = 0
        for i, g in enumerate(self.games):
            curve = g.get("eval_curve", [])
            moves = g.get("move_list", [])
            result = g.get("result", "loss")
            move_str = " ".join(
                f"{m // 2 + 1}.{moves[m]}"
                + (f" {moves[m+1]}" if m + 1 < len(moves) else "")
                for m in range(0, len(moves), 2)
            )
            blunder_details = []
            for j in range(1, len(curve)):
                drop = curve[j] - curve[j - 1]
                if drop < -200:
                    move_num = j // 2 + 1
                    move_text = moves[j] if j < len(moves) else "?"
                    blunder_details.append(
                        f"move {move_num} ({move_text}): "
                        f"{curve[j-1]:+d}cp -> {curve[j]:+d}cp "
                        f"(drop {drop:+d}cp)"
                    )
            won = result == "win"
            drew = result == "draw"
            if i == 0:
                for role in ["tactician", "positionalist", "selector", "verifier"]:
                    steps.append(AgentStep(
                        order=order, role=role, started_at="",
                        duration_s=0, cost_usd=None, output_tokens=None,
                        succeeded=True, node_id=f"game{i}_{role}",
                    ))
                    order += 1
            curve_str = ",".join(f"{c:+d}" for c in curve[-10:])
            avg = sum(curve) / len(curve) if curve else 0
            blunder_str = (
                "; ".join(blunder_details[:3])
                if blunder_details else "none"
            )
            agent_out = g.get("agent_outputs", {})
            sel_list = agent_out.get("selector", [])
            ver_list = agent_out.get("verifier", [])
            selector_out = " | ".join(
                sel_list[-3:] if isinstance(sel_list, list) else [sel_list]
            )
            verifier_out = " | ".join(
                ver_list[-3:] if isinstance(ver_list, list) else [ver_list]
            )
            hypothesis = (
                f"{g.get('tag', '')} | {result} in {len(moves)} moves | "
                f"moves: {move_str[:200]} | "
                f"eval: [{curve_str}] | "
                f"blunders ({len(blunder_details)}): {blunder_str}"
            )
            if selector_out:
                hypothesis += f" | selector: {selector_out}"
            if verifier_out:
                hypothesis += f" | verifier: {verifier_out}"
            experiments.append(ExperimentRecord(
                exp_id=i, hypothesis=hypothesis,
                verdict="keep" if won or drew else "revert",
                score_before=0, score_after=avg,
                score_delta=avg, cost_usd=0, duration_s=0,
            ))
        curves = []
        for g in self.games:
            curves.extend(g.get("eval_curve", []))
        return CycleRecord(
            cycle_number=gen, mode="chess", started_at=None, ended_at=None,
            duration_s=0,
            score_start=0, score_end=self.composite_score,
            score_delta=self.composite_score,
            score_trajectory=[float(c) for c in curves],
            experiments=experiments,
            kept=self.wins + self.draws, reverted=self.losses,
            errored=self.total_errors,
            keep_rate=(self.wins + self.draws) / max(self.wins + self.draws + self.losses, 1),
            steps=steps,
        )

    @property
    def win_rate(self) -> str:
        return f"+{self.wins}={self.draws}-{self.losses}"


async def play_game(
    pipeline: Package,
    cfg: PipelineConfig,
    llm_plays_white: bool = True,
    game_tag: str = "",
    gen: int = 0,
) -> dict:
    """Play one game using the pipeline for each LLM move."""
    safe_tag = game_tag.replace(":", "_").replace("(", "").replace(")", "").replace(" ", "_")[:80]
    workspace = WORKSPACE / safe_tag
    setup_workspace(workspace)

    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": cfg.opponent_elo})
    evaluator = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    move_count = 0
    llm_errors = 0
    pipeline_runs = 0
    game_moves: list[str] = []
    eval_curve: list[int] = []
    current_plan: str = ""
    all_node_outputs: dict[str, list[str]] = {}

    try:
        while not board.is_game_over() and move_count < MAX_MOVES:
            is_llm_turn = (board.turn == chess.WHITE) == llm_plays_white

            if is_llm_turn:
                legal = list(board.legal_moves)

                if cfg.skip_forced and len(legal) <= 2:
                    move_uci = legal[0].uci()
                    board.push(legal[0])
                    game_moves.append(move_uci)
                    move_count += 1
                    try:
                        info = evaluator.analyse(board, chess.engine.Limit(time=0.05))
                        cp = info["score"].white().score(mate_score=10000)
                        eval_curve.append(cp if llm_plays_white else -cp)
                    except Exception:
                        eval_curve.append(0)
                    broadcast_game_state(
                        game_tag, board, game_moves, llm_plays_white,
                        cfg.opponent_elo, result=None, move_count=move_count,
                        gen=gen, config=cfg.label, eval_curve=eval_curve,
                        full_config=cfg.full_label,
                    )
                    print(f"      {DIM}[{game_tag}] {format_moves(game_moves)} (forced){RESET}")
                    continue

                opponent_context = ""
                if cfg.use_opponent_model:
                    user_msg = _board_user_msg(board, game_moves, use_context=True)
                    opponent_context = await _call_llm(
                        OPPONENT_MODEL_PROMPT, user_msg, max_tokens=150,
                    )

                effective_cfg = cfg

                extra_context = ""
                if current_plan:
                    extra_context += f"\nCurrent plan: {current_plan}"
                if opponent_context:
                    extra_context += f"\nOpponent threats: {opponent_context}"

                try:
                    move_uci, nodes_executed, node_outputs = await asyncio.wait_for(
                        get_pipeline_move(
                            pipeline, board, effective_cfg, workspace,
                            game_tag=game_tag, llm_white=llm_plays_white,
                            stockfish_elo=cfg.opponent_elo, game_moves=game_moves,
                            move_count=move_count, gen=gen, config_label=cfg.label,
                            full_config_label=cfg.full_label,
                            eval_curve=eval_curve, extra_context=extra_context,
                            accumulated_outputs=all_node_outputs,
                        ),
                        timeout=300.0,
                    )
                except asyncio.TimeoutError:
                    move_uci = None
                pipeline_runs += 1

                source = (
                    node_outputs.get("_move_source", "none")
                    if node_outputs else "none"
                )
                if move_uci is None:
                    llm_errors += 1
                    move = random.choice(legal)
                    move_uci = move.uci()
                    source = "random"

                import sys
                selector_said = (
                    node_outputs.get("selector", "?")[:20]
                    if node_outputs else "?"
                )
                print(
                    f"  [{game_tag}] move {move_count+1}"
                    f" played={move_uci}"
                    f" selector={selector_said}"
                    f" source={source}",
                    file=sys.stderr, flush=True,
                )

                board.push_uci(move_uci)

                if cfg.use_game_plan:
                    plan_input = _board_user_msg(board, game_moves, use_context=True)
                    if current_plan:
                        plan_input += f"\nPrevious plan: {current_plan}"
                    current_plan = await _call_llm(GAME_PLAN_PROMPT, plan_input, max_tokens=60)
            else:
                broadcast_game_state(
                    game_tag, board, game_moves, llm_plays_white,
                    cfg.opponent_elo, move_count=move_count,
                    gen=gen, config=cfg.label,
                    whose_turn="stockfish", active_node="",
                    eval_curve=eval_curve, full_config=cfg.full_label,
                )
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move_uci = result.move.uci()
                board.push(result.move)

            game_moves.append(move_uci)
            move_count += 1

            try:
                info = evaluator.analyse(board, chess.engine.Limit(time=0.05))
                score = info["score"].white()
                cp = score.score(mate_score=10000)
                eval_cp = cp if llm_plays_white else -cp
            except Exception:
                eval_cp = 0
            eval_curve.append(eval_cp)

            broadcast_game_state(
                game_tag, board, game_moves, llm_plays_white,
                cfg.opponent_elo, result=None, move_count=move_count,
                gen=gen, config=cfg.label,
                eval_curve=eval_curve, full_config=cfg.full_label,
            )
            print(f"      {DIM}[{game_tag}] {format_moves(game_moves)} eval={eval_cp:+d}cp{RESET}")

            if len(eval_curve) >= 2 and all(e < -500 for e in eval_curve[-2:]):
                print(f"      {DIM}[{game_tag}] Resigning (eval < -500cp for 2 moves){RESET}")
                break

        outcome = board.outcome()
        if outcome is not None and outcome.winner is not None:
            if (outcome.winner == chess.WHITE) == llm_plays_white:
                result_str = "win"
            else:
                result_str = "loss"
        elif outcome is not None and outcome.winner is None:
            result_str = "draw"
        else:
            final_eval = eval_curve[-1] if eval_curve else 0
            if final_eval > 100:
                result_str = "draw"
            elif final_eval < -100:
                result_str = "loss"
            else:
                result_str = "draw"

        broadcast_game_state(
            game_tag, board, game_moves, llm_plays_white,
            cfg.opponent_elo, result=result_str, move_count=move_count,
            gen=gen, config=cfg.label,
            eval_curve=eval_curve, full_config=cfg.full_label,
            node_outputs=all_node_outputs,
        )
    finally:
        engine.quit()
        evaluator.quit()

    return {
        "result": result_str, "score": 0, "moves": move_count,
        "llm_errors": llm_errors, "pipeline_runs": pipeline_runs,
        "termination": outcome.termination.name if outcome else "max_moves",
        "move_list": game_moves,
        "eval_curve": eval_curve,
        "agent_outputs": dict(all_node_outputs),
    }


async def evaluate_pipeline(
    pipeline: Package,
    cfg: PipelineConfig,
    n_games: int = GAMES_PER_EVAL,
    eval_tag: str = "",
    gen: int = 0,
) -> EvalResult:
    games = []
    for i in range(n_games):
        color = "W" if i % 2 == 0 else "B"
        game_tag = f"{eval_tag}:g{i+1}({color})"
        game = await play_game(
            pipeline, cfg, llm_plays_white=(i % 2 == 0),
            game_tag=game_tag, gen=gen,
        )
        games.append(game)

    result = EvalResult()
    for game in games:
        result.games.append(game)
        if game["result"] == "win":
            result.wins += 1
        elif game["result"] == "draw":
            result.draws += 1
        else:
            result.losses += 1
        result.total_moves += game["moves"]
        result.total_errors += game["llm_errors"]
        result.total_pipeline_runs += game["pipeline_runs"]

        tag = eval_tag or "?"
        print(f"    {WHITE}{tag}: {game['result'].upper()}{RESET} in {game['moves']} moves "
              f"vs SF {cfg.opponent_elo}  "
              f"({game['pipeline_runs']} pipelines, {game['llm_errors']} errors)")
        print(f"    {DIM}{format_moves(game['move_list'])}{RESET}")
    return result
