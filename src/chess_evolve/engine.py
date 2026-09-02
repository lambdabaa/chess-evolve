"""Board I/O, LLM calling, move extraction, and pipeline execution."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path

import chess
from factory.workflow.executor import WorkflowExecutor
from factory.workflow.package import Package

from chess_evolve.broadcast import broadcast_game_state
from chess_evolve.pipeline import PipelineConfig
from chess_evolve.prompts import detect_phase

_client = None
_api_semaphore = asyncio.Semaphore(2)
CHESS_MODEL = os.environ.get("CHESS_MODEL", "opus")
USE_HAIKU_API = os.environ.get("CHESS_USE_HAIKU_API", "").lower() in ("1", "true", "yes")
HAIKU_MODEL = os.environ.get("CHESS_HAIKU_MODEL", "claude-haiku-4-5@20251001")
MODEL_LABEL = "Haiku 4.5 (API)" if USE_HAIKU_API else f"CLI ({CHESS_MODEL})"

_node_reads_by_prompt: dict[str, set[str]] = {}


def _get_client():
    global _client
    if _client is None:
        from anthropic import AsyncAnthropicVertex
        _client = AsyncAnthropicVertex(region="us-east5")
    return _client


async def _haiku_api_call(
    system_prompt: str, user_msg: str, max_tokens: int = 200,
) -> str:
    """Call Haiku directly via Vertex API."""
    try:
        client = _get_client()
        response = await asyncio.wait_for(
            client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            ),
            timeout=60.0,
        )
        return response.content[0].text.strip() if response.content else ""
    except Exception:
        return ""


async def _cli_call(
    system_prompt: str, user_msg: str, max_tokens: int = 200,
) -> str:
    """Call LLM via claude CLI with retry."""
    import sys
    env = _clean_env()
    for attempt in range(3):
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p", user_msg,
                "--model", CHESS_MODEL,
                "--append-system-prompt", system_prompt,
                "--max-turns", "2",
                "--output-format", "text",
                "--bare",
                "--allowedTools", "",
                "--effort", "low",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd="/tmp",
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=45.0,
            )
            result = stdout.decode().strip()
            if "max turns" in result.lower() or "reached max" in result.lower():
                print(
                    f"  [CLI] HIT MAX TURNS (attempt {attempt+1})",
                    file=sys.stderr, flush=True,
                )
                await asyncio.sleep(2)
                continue
            if result:
                return result
            print(
                f"  [CLI] empty (attempt {attempt+1})",
                file=sys.stderr, flush=True,
            )
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            print(
                f"  [CLI] timeout (attempt {attempt+1})",
                file=sys.stderr, flush=True,
            )
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(1)
    return ""


async def _api_call(
    system_prompt: str, user_msg: str, max_tokens: int = 200,
) -> str:
    """Route to Haiku API or CLI based on CHESS_USE_HAIKU_API."""
    if USE_HAIKU_API:
        return await _haiku_api_call(system_prompt, user_msg, max_tokens)
    return await _cli_call(system_prompt, user_msg, max_tokens)


def _clean_env() -> dict[str, str]:
    """Strip Claude Code session vars so subprocesses start fresh."""
    skip = {
        "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_PID", "AI_AGENT", "CLAUDE_CODE_ENTRYPOINT",
    }
    return {k: v for k, v in os.environ.items() if k not in skip}


async def _cli_call_opus(
    system_prompt: str, user_msg: str, max_tokens: int = 500,
) -> str:
    """Call Opus via claude CLI with retry."""
    import sys
    env = _clean_env()
    for attempt in range(3):
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p", user_msg,
                "--model", "opus",
                "--append-system-prompt", system_prompt,
                "--max-turns", "1",
                "--output-format", "text",
                "--bare",
                "--allowedTools", "",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd="/tmp",
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=180.0,
            )
            result = stdout.decode().strip()
            err = stderr.decode().strip()
            if err:
                print(
                    f"  [OPUS] stderr (attempt {attempt+1}): {err[:200]}",
                    file=sys.stderr, flush=True,
                )
            if "max turns" in result.lower() or "reached max" in result.lower():
                print(
                    f"  [OPUS] HIT MAX TURNS (attempt {attempt+1}): "
                    f"stdout={result[:150]!r} returncode={proc.returncode}",
                    file=sys.stderr, flush=True,
                )
                continue
            if result:
                return result
            print(
                f"  [OPUS] empty response (attempt {attempt+1})"
                f" returncode={proc.returncode}",
                file=sys.stderr, flush=True,
            )
        except asyncio.TimeoutError:
            print(
                f"  [OPUS] timeout (attempt {attempt+1})",
                file=sys.stderr, flush=True,
            )
            try:
                proc.kill()
            except Exception:
                pass
        except Exception as exc:
            print(
                f"  [OPUS] error: {exc} (attempt {attempt+1})",
                file=sys.stderr, flush=True,
            )
    return ""


async def _sdk_invoke_agent(
    role: str,
    task: str,
    project_path: Path,
    model: str | None = None,
    timeout: float = 25.0,
    **kwargs: object,
) -> tuple[str, int]:
    """Drop-in for factory's invoke_agent using CLI."""
    system = (
        f"You are a chess {role}. "
        "RESPOND IN UNDER 100 WORDS. "
        "If selecting a move, output ONLY the UCI move (e.g. e2e4)."
    )
    reads = {
        ".factory/chess/board_state.md",
        ".factory/chess/memory.md",
        ".factory/chess/analysis.md",
        ".factory/chess/verification.md",
    }
    context_parts = []
    for rpath in sorted(reads):
        fpath = project_path / rpath
        if fpath.exists():
            content = fpath.read_text().strip()
            if content:
                context_parts.append(f"--- {Path(rpath).name} ---\n{content}")
    if context_parts:
        task = task + "\n\n" + "\n\n".join(context_parts)
    async with _api_semaphore:
        text = await _api_call(system, task)
        if not text:
            text = ""
        return text, 0


async def _call_llm(
    system_prompt: str, user_msg: str, max_tokens: int = 200,
    model_override: str = "",
) -> str:
    """Call LLM via claude CLI."""
    async with _api_semaphore:
        return await _api_call(system_prompt, user_msg, max_tokens)


def setup_workspace(workspace: Path) -> None:
    """Create the workspace with .factory/chess/ directory."""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / ".factory" / "chess").mkdir(parents=True)
    (workspace / ".factory" / "strategy").mkdir(parents=True)
    (workspace / ".factory" / "reviews").mkdir(parents=True)


def write_board_state(workspace: Path, board: chess.Board) -> None:
    """Write the current board state as an artifact for the pipeline to read."""
    legal_moves = [move.uci() for move in board.legal_moves]
    content = (
        f"FEN: {board.fen()}\n\n"
        f"You are playing {'White' if board.turn else 'Black'}.\n\n"
        f"Legal moves: {', '.join(legal_moves)}\n\n"
        f"Board:\n{board}\n"
    )
    (workspace / ".factory" / "chess" / "board_state.md").write_text(content)


def read_move(workspace: Path, board: chess.Board) -> str | None:
    """Read the move chosen by the pipeline from the artifact file."""
    move_file = workspace / ".factory" / "chess" / "move.md"
    if not move_file.exists():
        return None
    content = move_file.read_text().strip()
    legal_moves = [m.uci() for m in board.legal_moves]
    for token in content.split():
        cleaned = token.strip(".,!()[]{}\"'`\n")
        if cleaned in legal_moves:
            return cleaned
    for match in re.finditer(r'[a-h][1-8][a-h][1-8][qrbn]?', content):
        if match.group() in legal_moves:
            return match.group()
    return None


def _ascii_board(board: chess.Board) -> str:
    """Render board as ASCII with coordinates."""
    symbols = {"R": "R", "N": "N", "B": "B", "Q": "Q", "K": "K", "P": "P",
               "r": "r", "n": "n", "b": "b", "q": "q", "k": "k", "p": "p"}
    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = f"{rank+1} "
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row += (symbols[piece.symbol()] if piece else ".") + " "
        lines.append(row + f"{rank+1}")
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def _board_user_msg(
    board: chess.Board, game_moves: list[str] | None = None,
    use_context: bool = False, cfg: PipelineConfig | None = None,
) -> str:
    legal_moves = [m.uci() for m in board.legal_moves]
    board_repr = cfg.board_representation if cfg else "fen"
    parts = []
    if board_repr in ("fen", "both"):
        parts.append(f"Position (FEN): {board.fen()}")
    if board_repr in ("ascii", "both"):
        parts.append(f"Board:\n{_ascii_board(board)}")
    parts.extend([
        f"You are {'White' if board.turn else 'Black'}.",
        f"Legal moves: {', '.join(legal_moves)}",
    ])
    if use_context and game_moves:
        move_pairs = []
        for i in range(0, len(game_moves), 2):
            num = i // 2 + 1
            w = game_moves[i]
            b = game_moves[i + 1] if i + 1 < len(game_moves) else ""
            move_pairs.append(f"{num}.{w} {b}".strip())
        parts.insert(1, f"Game so far: {' '.join(move_pairs)}")
        parts.insert(2, f"Move number: {len(game_moves) // 2 + 1}")
    return "\n".join(parts)


def _extract_move(text: str, board: chess.Board) -> str | None:
    legal_moves = [m.uci() for m in board.legal_moves]
    for token in text.split():
        cleaned = token.strip(".,!()[]{}\"'`\n")
        if cleaned in legal_moves:
            return cleaned
    for match in re.finditer(r'[a-h][1-8][a-h][1-8][qrbn]?', text):
        if match.group() in legal_moves:
            return match.group()
    for token in text.split():
        cleaned = token.strip(".,!()[]{}\"'`+#\n")
        try:
            move = board.parse_san(cleaned)
            return move.uci()
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            pass
    return None


async def get_pipeline_move(
    pipeline: Package,
    board: chess.Board,
    cfg: PipelineConfig,
    workspace: Path,
    game_tag: str = "",
    llm_white: bool = True,
    stockfish_elo: int = 1320,
    game_moves: list[str] | None = None,
    move_count: int = 0,
    gen: int = 0,
    eval_curve: list[int] | None = None,
    config_label: str = "",
    full_config_label: str = "",
    extra_context: str = "",
    accumulated_outputs: dict[str, list[str]] | None = None,
) -> tuple[str | None, int, dict[str, str]]:
    """Run the Package pipeline through factory's real WorkflowExecutor."""
    user_msg = _board_user_msg(board, game_moves, use_context=cfg.use_game_context, cfg=cfg)
    if extra_context:
        user_msg += f"\n{extra_context}"
    chess_dir = workspace / ".factory" / "chess"
    chess_dir.mkdir(parents=True, exist_ok=True)
    (chess_dir / "board_state.md").write_text(user_msg)

    # Write rolling memory from previous moves' reasoning
    memory_path = chess_dir / "memory.md"
    if accumulated_outputs:
        memory_lines = []
        gen_history = accumulated_outputs.get("generator", [])
        # Keep last 3 moves of context
        start = max(0, len(gen_history) - 3)
        for i in range(start, len(gen_history)):
            move_num = i + 1
            memory_lines.append(f"## Move {move_num}")
            memory_lines.append(f"{gen_history[i][:300]}")
            memory_lines.append("")
        if memory_lines:
            memory_path.write_text("\n".join(memory_lines))
        elif memory_path.exists():
            memory_path.unlink()
    elif memory_path.exists():
        memory_path.unlink()

    for name in ["analysis.md", "verification.md"]:
        f = chess_dir / name
        if f.exists():
            f.unlink()
    fallback_path = workspace / ".factory" / "reviews" / "strategist-latest.md"
    if fallback_path.exists():
        fallback_path.unlink()

    wf = pipeline.compile()
    node_outputs: dict[str, str] = {}

    _node_reads_by_prompt.clear()
    for nid, node in wf.nodes.items():
        if hasattr(node, 'prompt_template') and hasattr(node, 'reads') and node.reads:
            _node_reads_by_prompt[nid] = node.reads

    _executor_holder: dict[str, WorkflowExecutor] = {}

    def make_hooked_emit(original_emit):
        NODE_NAME_MAP = {
            "generator": "generator",
            "legality_gate": "legality",
        }

        def _hooked_emit(event_type, event):
            original_emit(event_type, event)
            if not game_tag:
                return
            node_id = getattr(event, "node_id", "")
            if event_type == "node.started":
                active = NODE_NAME_MAP.get(node_id, node_id)
                broadcast_game_state(
                    game_tag, board, game_moves or [], llm_white,
                    stockfish_elo, move_count=move_count,
                    gen=gen, config=config_label,
                    whose_turn="llm", active_node=active,
                    eval_curve=eval_curve, full_config=full_config_label,
                )
            elif event_type == "node.completed":
                ex = _executor_holder.get("ex")
                if ex and event.node_id in ex.result.node_outputs:
                    output = ex.result.node_outputs[event.node_id]
                    node_outputs[event.node_id] = output
                    if accumulated_outputs is not None:
                        accumulated_outputs.setdefault(
                            event.node_id, [],
                        ).append(output)
                    broadcast_game_state(
                        game_tag, board, game_moves or [], llm_white,
                        stockfish_elo, move_count=move_count,
                        gen=gen, config=config_label,
                        whose_turn="llm",
                        active_node=NODE_NAME_MAP.get(
                            event.node_id, event.node_id,
                        ),
                        node_outputs=(
                            accumulated_outputs
                            if accumulated_outputs is not None
                            else node_outputs
                        ),
                        eval_curve=eval_curve,
                        full_config=full_config_label,
                    )
                    node_def = wf.nodes.get(event.node_id)
                    if node_def and node_def.writes and output:
                        for wpath in node_def.writes:
                            fpath = workspace / wpath
                            fpath.parent.mkdir(parents=True, exist_ok=True)
                            fpath.write_text(output)
        return _hooked_emit

    import factory.agents.runner as _runner
    _orig_invoke = _runner.invoke_agent
    _runner.invoke_agent = _sdk_invoke_agent  # type: ignore[assignment]
    try:
        executor = WorkflowExecutor(
            workflow=wf, project_path=workspace, auto_approve=True,
        )
        _executor_holder["ex"] = executor
        executor.completed_files.add(".factory/chess/board_state.md")
        executor.completed_files.add(".factory/chess/memory.md")
        executor._emit = make_hooked_emit(executor._emit)  # type: ignore[assignment]
        result = await executor.execute()
    finally:
        _runner.invoke_agent = _orig_invoke  # type: ignore[assignment]

    import sys
    move_tag = game_tag or "?"
    print(
        f"  [{move_tag}] EXEC outputs={list(result.node_outputs.keys())}"
        f" halted={result.halted}",
        file=sys.stderr, flush=True,
    )
    for nid, output in result.node_outputs.items():
        node = wf.nodes.get(nid)
        if node and node.writes and output:
            for wpath in node.writes:
                fpath = workspace / wpath
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(output)
                print(
                    f"  [{move_tag}] WRITE {wpath} ({len(output)} chars)",
                    file=sys.stderr, flush=True,
                )

    # Find the move from whichever node writes move.md
    move = None
    move_source = "none"
    move_node = None
    for nid, node in wf.nodes.items():
        if hasattr(node, "writes") and ".factory/chess/move.md" in (node.writes or set()):
            move_node = nid
            break
    if move_node and move_node in result.node_outputs:
        output = result.node_outputs[move_node]
        move = _extract_move(output, board)
        if move:
            move_source = "pipeline"
        else:
            print(
                f"  [{move_tag}] PARSE_FAIL {move_node}={output[:80]!r}",
                file=sys.stderr, flush=True,
            )
    # Fallback: try every node output for a legal move
    if move is None:
        for nid, output in result.node_outputs.items():
            move = _extract_move(output, board)
            if move:
                move_source = f"{nid}_fallback"
                break

    node_outputs["_move_source"] = move_source
    return move, result.nodes_executed, node_outputs
