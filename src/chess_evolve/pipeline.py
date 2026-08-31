"""Pipeline definition: PipelineConfig + build_pipeline() using factory's Package ecosystem."""

from __future__ import annotations

from dataclasses import dataclass

from factory.workflow.package import (
    Conditional,
    Loop,
    OptKnob,
    Package,
    Parallel,
    Port,
    Sequential,
    StateContract,
)
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    GateNode,
    Workflow,
)

from chess_evolve.prompts import (
    ANALYSIS_PROMPTS,
    SELECTOR_PROMPTS,
    _get_prompt,
)

KNOB_SPACE: list[tuple[str, list]] = [
    ("verify_style", ["strict", "standard", "lenient"]),
    ("verify_iterations", [1, 2, 3]),
    ("critique_style", ["devils_advocate", "sanity_check"]),
    ("use_verification", [True, False]),
    ("opening_hint", ["theory", "principled"]),
]

_PROMPT_NODES = ["board_analyst", "tactician", "positionalist", "selector", "verifier"]


@dataclass
class PipelineConfig:
    """Tunable knobs for the chess pipeline."""

    # Fixed
    model: str = "haiku"
    think_tokens: int = 200
    selector_style: str = "balanced"
    analysis_style: str = "concise"
    include_analyst: bool = True
    opponent_elo: int = 1320
    endgame_threshold: int = 25

    # Fixed (Opus found these always hurt or don't matter)
    use_game_context: bool = False
    use_game_plan: bool = False
    selector_tokens: int = 10
    verify_loop_target: str = "selector_only"
    verify_iterations: int = 2

    # Locked winners (from 8000 experiments)
    pipeline_mode: str = "parallel"
    tactical_style: str = "capture_refutation_gate"
    positional_style: str = "prophylactic"
    middlegame_hint: str = "theory+safety_first"
    endgame_hint: str = "theory+technical+convert"
    board_representation: str = "both"
    game_phase_routing: bool = False
    skip_forced: bool = False
    use_opponent_model: bool = False

    # Tunable
    use_verification: bool = True
    verify_style: str = "strict"
    critique_style: str = "sanity_check"
    opening_hint: str = "theory"

    @property
    def label(self) -> str:
        parts = [f"mode={self.pipeline_mode}"]
        if self.tactical_style != "mate_focused":
            parts.append(f"tact={self.tactical_style}")
        if self.positional_style != "dynamic":
            parts.append(f"pos={self.positional_style}")
        if self.verify_style != "strict":
            parts.append(f"verify={self.verify_style}")
        if self.critique_style != "sanity_check":
            parts.append(f"crit={self.critique_style}")
        if self.verify_loop_target != "selector_only":
            parts.append("reanalyze")
        if not self.use_verification:
            parts.append("no-verify")
        elif self.verify_iterations != 2:
            parts.append(f"verify-x{self.verify_iterations}")
        if self.selector_tokens != 10:
            parts.append(f"tok={self.selector_tokens}")
        if self.use_game_context:
            parts.append("ctx")
        if self.use_game_plan:
            parts.append("plan")
        if self.use_opponent_model:
            parts.append("opp")
        if self.skip_forced:
            parts.append("skip")
        if self.board_representation != "both":
            parts.append(f"repr={self.board_representation}")
        if self.game_phase_routing:
            parts.append("phased")
        if self.opening_hint != "theory":
            parts.append(f"open={self.opening_hint}")
        if self.middlegame_hint != "theory":
            parts.append(f"mid={self.middlegame_hint}")
        if self.endgame_hint != "theory":
            parts.append(f"end={self.endgame_hint}")
        return " ".join(parts)

    @property
    def full_label(self) -> str:
        return " | ".join(
            f"{k}={getattr(self, k)}" for k, _ in KNOB_SPACE if hasattr(self, k)
        )


def build_pipeline(cfg: PipelineConfig | None = None) -> Package:
    """Build the chess move pipeline as a Package composition."""
    if cfg is None:
        cfg = PipelineConfig()

    tactician = AgentNode(
        id="tactician", role=AgentRole.RESEARCHER,
        prompt_template=_get_prompt("tactical_style", cfg.tactical_style),
        reads={".factory/chess/board_state.md"},
        writes={".factory/chess/tactics.md"},
    )
    tactical_pkg = Package(
        name="tactical-scan", version="1.0.0",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="tactics", artifact_path=".factory/chess/tactics.md")],
        contract=StateContract(produces=frozenset({"tactics_complete"}),
                               capabilities=["chess-tactics"]),
        graph=Workflow(name="tactical-scan", nodes={"tactician": tactician},
                       edges=[], start_node="tactician"),
        entry_node="tactician", exit_node="tactician",
    )

    positionalist = AgentNode(
        id="positionalist", role=AgentRole.RESEARCHER,
        prompt_template=_get_prompt("positional_style", cfg.positional_style),
        reads={".factory/chess/board_state.md"},
        writes={".factory/chess/positional.md"},
    )
    positional_pkg = Package(
        name="positional-eval", version="1.0.0",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="positional", artifact_path=".factory/chess/positional.md")],
        contract=StateContract(produces=frozenset({"positional_complete"}),
                               capabilities=["chess-positional"]),
        graph=Workflow(name="positional-eval", nodes={"positionalist": positionalist},
                       edges=[], start_node="positionalist"),
        entry_node="positionalist", exit_node="positionalist",
    )

    selector = AgentNode(
        id="selector", role=AgentRole.STRATEGIST,
        prompt_template=SELECTOR_PROMPTS[cfg.selector_style],
        reads={".factory/chess/tactics.md", ".factory/chess/positional.md",
               ".factory/chess/board_state.md"},
        writes={".factory/chess/move.md"},
    )
    selector_pkg = Package(
        name="move-selector", version="1.0.0",
        inputs=[Port(name="tactics", artifact_path=".factory/chess/tactics.md"),
                Port(name="positional", artifact_path=".factory/chess/positional.md")],
        outputs=[Port(name="move", artifact_path=".factory/chess/move.md")],
        contract=StateContract(
            requires=frozenset({"tactics_complete", "positional_complete"}),
            produces=frozenset({"move_selected"}),
            capabilities=["chess-move-selection"]),
        graph=Workflow(name="move-selector", nodes={"selector": selector},
                       edges=[], start_node="selector"),
        entry_node="selector", exit_node="selector",
    )

    verify_prompt_parts = [_get_prompt("critique_style", cfg.critique_style),
                           _get_prompt("verify_style", cfg.verify_style)]
    verify_prompt = " ".join(verify_prompt_parts)

    verifier = AgentNode(
        id="verifier", role=AgentRole.RESEARCHER,
        prompt_template=verify_prompt,
        reads={".factory/chess/move.md", ".factory/chess/board_state.md"},
        writes={".factory/chess/verification.md"},
    )
    verify_pkg = Package(
        name="move-verify", version="1.0.0",
        inputs=[Port(name="move", artifact_path=".factory/chess/move.md")],
        outputs=[Port(name="verify", artifact_path=".factory/chess/verification.md")],
        contract=StateContract(requires=frozenset({"move_selected"}),
                               produces=frozenset({"move_verified"})),
        graph=Workflow(name="move-verify", nodes={"verifier": verifier},
                       edges=[], start_node="verifier"),
        entry_node="verifier", exit_node="verifier",
    )

    verify_gate = GateNode(
        id="verify_gate",
        evaluator_type="fn",
        evaluator_command=(
            'python3 -c "'
            "from pathlib import Path; "
            "text = Path('{project_path}/.factory/chess/verification.md').read_text().lower() "
            "if Path('{project_path}/.factory/chess/verification.md').exists() else ''; "
            "has_issue = 'blunder' in text or "
            "('objection' in text and 'no objection' not in text); "
            'print("RELOOP" if has_issue else "PROCEED")'
            '"'
        ),
    )

    parallel_analysis = Parallel(tactical_pkg, positional_pkg, name="parallel-analysis")

    analyst = AgentNode(
        id="board_analyst", role=AgentRole.RESEARCHER,
        prompt_template=ANALYSIS_PROMPTS[cfg.analysis_style],
        reads={".factory/chess/board_state.md"},
        writes={".factory/chess/analysis.md"},
    )
    analyst_pkg = Package(
        name="board-analysis", version="1.0.0",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="analysis", artifact_path=".factory/chess/analysis.md")],
        contract=StateContract(produces=frozenset({"analysis_complete"}),
                               capabilities=["chess-analysis"]),
        graph=Workflow(name="board-analysis", nodes={"board_analyst": analyst},
                       edges=[], start_node="board_analyst"),
        entry_node="board_analyst", exit_node="board_analyst",
    )
    if cfg.include_analyst:
        parallel_analysis = Parallel(
            analyst_pkg, tactical_pkg, positional_pkg, name="parallel-analysis"
        )

    if cfg.game_phase_routing:
        opening_analyst = AgentNode(
            id="opening_analyst", role=AgentRole.RESEARCHER,
            prompt_template=(
                "This is a chess OPENING position. "
                "Play principled opening moves: center pawns (e4/d4 or e5/d5/c5), "
                "develop knights (Nf3/Nc3 or Nf6/Nc6), then bishops, then castle. "
                "NEVER play a3, a4, h3, h4, or edge pawn moves. "
                "Suggest the single best developing move in UCI notation."
            ),
            reads={".factory/chess/board_state.md"},
            writes={".factory/chess/analysis.md"},
        )
        opening_pkg = Package(
            name="opening-analysis", version="1.0.0",
            inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
            outputs=[Port(name="analysis", artifact_path=".factory/chess/analysis.md")],
            contract=StateContract(produces=frozenset({"analysis_complete"})),
            graph=Workflow(name="opening-analysis", nodes={"opening_analyst": opening_analyst},
                           edges=[], start_node="opening_analyst"),
            entry_node="opening_analyst", exit_node="opening_analyst",
        )
        endgame_analyst = AgentNode(
            id="endgame_analyst", role=AgentRole.RESEARCHER,
            prompt_template=(
                "This is a chess ENDGAME position. Focus on: "
                "1) King activity -- can your king advance safely? "
                "2) Passed pawns -- create or push them. "
                "3) Piece trades -- trade if ahead, avoid if behind. "
                "Name the best endgame move in UCI."
            ),
            reads={".factory/chess/board_state.md"},
            writes={".factory/chess/analysis.md"},
        )
        endgame_pkg = Package(
            name="endgame-analysis", version="1.0.0",
            inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
            outputs=[Port(name="analysis", artifact_path=".factory/chess/analysis.md")],
            contract=StateContract(produces=frozenset({"analysis_complete"})),
            graph=Workflow(name="endgame-analysis", nodes={"endgame_analyst": endgame_analyst},
                           edges=[], start_node="endgame_analyst"),
            entry_node="endgame_analyst", exit_node="endgame_analyst",
        )
        phase_gate = GateNode(
            id="phase_gate",
            evaluator_type="fn",
            evaluator_command=(
                'python3 -c "'
                "fen = open('{project_path}/.factory/chess/"
                "board_state.md').readline().split(': ')[1].strip(); "
                "pieces = sum(1 for c in fen.split()[0] if c.isalpha() and c.lower() != 'k'); "
                'print("HALT" if pieces > 26 else "RELOOP" if pieces <= 10 else "PROCEED")'
                '"'
            ),
        )
        analysis_step = Conditional(
            phase_gate,
            {"HALT": opening_pkg, "PROCEED": parallel_analysis, "RELOOP": endgame_pkg},
            name="phase-router",
        )
    else:
        analysis_step = parallel_analysis

    selector.reads.add(".factory/chess/verification.md")
    if cfg.use_verification:
        if cfg.verify_loop_target == "full_reanalysis":
            loop_body = Sequential(analysis_step, selector_pkg, verify_pkg,
                                   name="analyze-select-verify")
        else:
            loop_body = Sequential(selector_pkg, verify_pkg, name="select-verify")
        select_and_verify = Loop(
            loop_body, verify_gate, max_iterations=cfg.verify_iterations, name="verify-loop",
        )
    else:
        select_and_verify = selector_pkg

    analysis_in_loop = (
        cfg.use_verification and cfg.verify_loop_target == "full_reanalysis"
    )

    if analysis_in_loop:
        result = select_and_verify
    else:
        result = Sequential(analysis_step, select_and_verify, name="chess-pipeline")

    knobs = []
    for knob_name, choices in KNOB_SPACE:
        val = getattr(cfg, knob_name, None)
        if val is None:
            continue

        def coerce(v: object) -> str | float:
            return str(v) if isinstance(v, bool) else v  # type: ignore[return-value]

        kind = ("prompt" if isinstance(val, str)
                and knob_name not in ("pipeline_mode", "verify_loop_target", "board_representation")
                else "threshold" if isinstance(val, (int, float)) else "topology")
        knobs.append(OptKnob(
            name=knob_name, kind=kind, node_id="root",
            default=coerce(val), bounds=[coerce(c) for c in choices],
            expandable=kind == "prompt",
            expansion_hint=f"chess {knob_name} variant",
        ))
    return result.model_copy(update={"knobs": knobs})
