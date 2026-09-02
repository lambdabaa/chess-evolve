"""Pipeline definition: PipelineConfig, KNOB_SPACE, build_pipeline().

Minimal starting pipeline — a generic move generator with a legality
check loop. The optimizer discovers chess strategy from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

from factory.workflow.package import (
    Loop,
    MemoryDeclaration,
    OptKnob,
    Package,
    Port,
    Sequential,
    StateContract,
)
from factory.workflow.primitives import AgentNode, AgentRole, GateNode, Workflow

KNOB_SPACE: list[tuple[str, list]] = [
    ("max_retries", [1, 2, 3, 5]),
    ("generator_style", ["generic"]),
]

_PROMPT_NODES = ["generator"]

GENERATOR_PROMPT = (
    "You are playing chess. Look at the board position and legal moves. "
    "Pick a move. Output ONLY the UCI move (e.g. e2e4). Nothing else."
)


@dataclass
class PipelineConfig:
    """Tunable knobs for the chess pipeline."""

    opponent_elo: int = 1500
    board_representation: str = "both"
    skip_forced: bool = False
    use_game_context: bool = False

    # Tunable
    max_retries: int = 3
    generator_style: str = "generic"

    @property
    def label(self) -> str:
        defaults = PipelineConfig()
        parts = []
        for knob_name, _ in KNOB_SPACE:
            val = getattr(self, knob_name)
            default_val = getattr(defaults, knob_name)
            if val != default_val:
                parts.append(f"{knob_name}={val}")
        return " ".join(parts) if parts else "seed"

    @property
    def full_label(self) -> str:
        return " | ".join(
            f"{k}={getattr(self, k)}" for k, _ in KNOB_SPACE if hasattr(self, k)
        )


def build_pipeline(cfg: PipelineConfig | None = None) -> Package:
    """Build a minimal chess pipeline: generator → legality gate → loop.

    The optimizer can add nodes, rewrite prompts, and change topology.
    """
    if cfg is None:
        cfg = PipelineConfig()

    from chess_evolve.prompts import _get_prompt
    prompt = _get_prompt("generator_style", cfg.generator_style)
    if not prompt or prompt == cfg.generator_style:
        prompt = GENERATOR_PROMPT

    generator = AgentNode(
        id="generator", role=AgentRole.STRATEGIST,
        prompt_template=prompt,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/chess/move.md"},
    )
    generator_pkg = Package(
        name="move-generator", version="1.0.0",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="move", artifact_path=".factory/chess/move.md")],
        contract=StateContract(produces=frozenset({"move_generated"})),
        graph=Workflow(name="move-generator", nodes={"generator": generator},
                       edges=[], start_node="generator"),
        entry_node="generator", exit_node="generator",
        memory=[MemoryDeclaration(
            namespace="game_reasoning",
            kind="log",
            schema_def={"move": "int", "generator": "str"},
            retention="ephemeral",
        )],
    )

    legality_gate = GateNode(
        id="legality_gate",
        evaluator_type="fn",
        evaluator_command=(
            "python3 -c '"
            "from pathlib import Path; "
            "import chess; "
            "board_text = Path(\"{project_path}/.factory/chess/"
            "board_state.md\").read_text(); "
            "fen_line = [l for l in board_text.split(chr(10)) "
            "if l.startswith(\"Position\") or l.startswith(\"FEN\")]; "
            "fen = fen_line[0].split(\": \", 1)[1].strip() if fen_line "
            "else board_text.split(chr(10))[0].split(\": \", 1)[-1].strip(); "
            "board = chess.Board(fen); "
            "legal = [m.uci() for m in board.legal_moves]; "
            "move_text = Path(\"{project_path}/.factory/chess/"
            "move.md\").read_text().strip(); "
            "tokens = move_text.replace(chr(10), \" \").split(); "
            "found = next((t.strip(\".,!()\\\"\\x27\") for t in tokens "
            "if t.strip(\".,!()\\\"\\x27\") in legal), None); "
            "print(\"PROCEED\" if found else \"RELOOP\")"
            "'"
        ),
    )

    loop_body = Sequential(generator_pkg, name="generate-move")
    move_loop = Loop(
        loop_body, legality_gate,
        max_iterations=cfg.max_retries, name="move-loop",
    )

    knobs = []
    for knob_name, choices in KNOB_SPACE:
        val = getattr(cfg, knob_name, None)
        if val is None:
            continue

        def coerce(v: object) -> str | float:
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, int):
                return float(v)
            return str(v)

        knobs.append(OptKnob(
            name=knob_name,
            kind="threshold" if knob_name == "max_retries" else "prompt",
            node_id="generator",
            default=coerce(val),
            bounds=[coerce(c) for c in choices],
            expansion_hint=(
                f"Max retry attempts for legal move generation"
                if knob_name == "max_retries"
                else f"Prompt style for the move generator"
            ),
        ))

    move_loop.knobs = knobs
    return move_loop
