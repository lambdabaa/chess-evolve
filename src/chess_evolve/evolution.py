"""Outer loop: evolutionary prompt optimization using factory's MAP-Elites."""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
import time
from dataclasses import dataclass, field

from factory.workflow.package import Package

from chess_evolve.broadcast import broadcast_archive, broadcast_eval_result
from chess_evolve.config import CANDIDATES_PER_GEN, GAMES_PER_EVAL, LIVE_DIR
from chess_evolve.display import CYAN, DIM, GREEN, MAGENTA, RESET, WHITE, header, print
from chess_evolve.engine import _cli_call_opus
from chess_evolve.game import EvalResult, evaluate_pipeline
from chess_evolve.pipeline import KNOB_SPACE, PipelineConfig, build_pipeline
from chess_evolve.prompts import PROMPT_REGISTRY, _register_prompt


@dataclass
class LeaderboardEntry:
    id: str
    label: str
    cfg: PipelineConfig
    pipeline: Package
    score: float
    gen: int
    prompts: dict[str, str] = field(default_factory=dict)


def mutate_knobs(
    cfg: PipelineConfig, rng: random.Random,
) -> tuple[PipelineConfig, str]:
    """Mutate exactly 1 knob."""
    new = dataclasses.replace(cfg)
    knob_name, choices = rng.choice(KNOB_SPACE)
    alternatives = [v for v in choices if v != getattr(new, knob_name)]
    if not alternatives:
        return new, "no-op"
    new_val = rng.choice(alternatives)
    object.__setattr__(new, knob_name, new_val)
    return new, f"{knob_name}={new_val}"


def _extract_json_objects(raw: str) -> list[dict]:
    """Extract JSON objects from Opus output, handling multi-line and arrays."""
    # Try as a JSON array first
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    # Extract individual JSON objects using brace matching
    objects = []
    i = 0
    while i < len(stripped):
        if stripped[i] == "{":
            depth = 0
            start = i
            in_string = False
            escape = False
            for j in range(i, len(stripped)):
                c = stripped[j]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(stripped[start:j + 1])
                                if isinstance(obj, dict):
                                    objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
            else:
                i += 1
        else:
            i += 1
    return objects


def _build_pipeline_from_spec(
    nodes_spec: list[dict], cfg: PipelineConfig,
) -> Package:
    """Build a pipeline from an Opus-designed node spec."""
    from factory.workflow.package import (
        Loop,
        MemoryDeclaration,
        Package,
        Port,
        Sequential,
        StateContract,
    )
    from factory.workflow.primitives import AgentNode, AgentRole, GateNode, Workflow

    pkgs = []
    for spec in nodes_spec:
        nid = spec["id"]
        prompt = spec.get("prompt", "")
        reads = set(spec.get("reads", [".factory/chess/board_state.md"]))
        reads.add(".factory/chess/memory.md")
        writes_path = spec.get("writes", f".factory/chess/{nid}.md")
        node = AgentNode(
            id=nid, role=AgentRole.RESEARCHER,
            prompt_template=prompt,
            reads=reads,
            writes={writes_path},
        )
        pkg = Package(
            name=nid, version="1.0.0",
            inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
            outputs=[Port(name=nid, artifact_path=writes_path)],
            contract=StateContract(produces=frozenset({f"{nid}_complete"})),
            graph=Workflow(name=nid, nodes={nid: node}, edges=[], start_node=nid),
            entry_node=nid, exit_node=nid,
        )
        pkgs.append(pkg)

    if not pkgs:
        return build_pipeline(cfg)

    # Add memory declaration to the first package
    pkgs[0] = pkgs[0].model_copy(update={"memory": [MemoryDeclaration(
        namespace="game_reasoning", kind="log",
        schema_def={"move": "int", "output": "str"},
        retention="ephemeral",
    )]})

    if len(pkgs) == 1:
        body = pkgs[0]
    else:
        body = Sequential(*pkgs, name="chess-agents")

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

    return Loop(body, legality_gate, max_iterations=cfg.max_retries, name="move-loop")


def parse_opus_proposals(
    raw: str, best_cfg: PipelineConfig, gen: int,
) -> list[tuple[str, PipelineConfig, object, int, dict]]:
    """Parse Opus JSON proposals into (label, cfg, pipeline, n_games, meta) tuples."""
    candidates: list[tuple[str, PipelineConfig, object, int, dict]] = []
    for prop in _extract_json_objects(raw):
        try:
            n_games = prop.get("games", GAMES_PER_EVAL)
            prompt_meta: dict = {}
            child_cfg = dataclasses.replace(best_cfg)

            if "pipeline" in prop and isinstance(prop["pipeline"], list):
                nodes_spec = prop["pipeline"]
                pipeline = _build_pipeline_from_spec(nodes_spec, child_cfg)
                node_names = [n["id"] for n in nodes_spec]
                desc = " → ".join(node_names)
                node_prompts = {
                    n["id"]: n.get("prompt", "")
                    for n in nodes_spec if n.get("prompt")
                }
                prompt_meta = {
                    "operator": "pipeline",
                    "nodes": node_names,
                    "node_count": len(node_names),
                    "node_prompts": node_prompts,
                }
                for n in nodes_spec:
                    if n.get("prompt"):
                        print(f"  {CYAN}NODE: {n['id']} ({len(n['prompt'])} chars){RESET}")
            elif "node" in prop and "prompt" in prop and "node" != "add_node":
                node = prop["node"]
                prompt_text = prop["prompt"]
                value = f"opus_g{gen}_{len(candidates)+1}"
                style_knob = f"{node}_style"
                _register_prompt(style_knob, value, prompt_text)
                for _, (kn, choices) in enumerate(KNOB_SPACE):
                    if kn == style_knob and value not in choices:
                        choices.append(value)
                if hasattr(child_cfg, style_knob):
                    child_cfg = dataclasses.replace(child_cfg, **{style_knob: value})
                pipeline = build_pipeline(child_cfg)
                desc = f"prompt({node})"
                prompt_meta = {"node": node, "after": prompt_text}
                print(f"  {CYAN}PROMPT: {node} -> {style_knob}={value}{RESET}")
            elif "knobs" in prop and isinstance(prop["knobs"], dict):
                overrides = {k: v for k, v in prop["knobs"].items() if hasattr(best_cfg, k)}
                for _, (kn, choices) in enumerate(KNOB_SPACE):
                    for k, v in overrides.items():
                        if kn == k and v not in choices:
                            choices.append(v)
                child_cfg = dataclasses.replace(best_cfg, **overrides)
                pipeline = build_pipeline(child_cfg)
                desc = " + ".join(f"{k}={v}" for k, v in overrides.items())
            else:
                continue

            candidates.append((
                f"Gen {gen}.{len(candidates)+1}: {desc}",
                child_cfg, pipeline, n_games, prompt_meta,
            ))
        except Exception:
            continue
    return candidates


def _load_invented_prompts() -> int:
    """Reload invented prompts from the recording on startup."""
    recording = LIVE_DIR / "recording.jsonl"
    if not recording.exists():
        return 0
    count = 0
    for line in recording.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            if e.get("type") == "prompt_invented":
                PROMPT_REGISTRY.setdefault(e["knob"], {})[e["value"]] = e["prompt"]
                for _, (kn, choices) in enumerate(KNOB_SPACE):
                    if kn == e["knob"] and e["value"] not in choices:
                        choices.append(e["value"])
                count += 1
        except Exception:
            continue
    return count


async def main():
    seed_cfg = PipelineConfig()
    seed = build_pipeline(seed_cfg)

    n_loaded = _load_invented_prompts()

    header("CHESS PIPELINE EVOLUTION -- adaptive loop")
    from chess_evolve.engine import MODEL_LABEL
    print(f"\n  {WHITE}Model:{RESET}       {MODEL_LABEL}")
    print(f"  {WHITE}Knobs:{RESET}       {len(KNOB_SPACE)}")
    if n_loaded:
        print(f"  {CYAN}Loaded:{RESET}      {n_loaded} invented prompts from recording")
    print(f"  {WHITE}Seed:{RESET}  {seed_cfg.label}")
    print(f"  {DIM}Compiled: {len(seed.compile().nodes)} nodes{RESET}")

    if LIVE_DIR.exists():
        (LIVE_DIR / "_reload.json").write_text(json.dumps({"_reload": True}))
        time.sleep(1)
        shutil.rmtree(LIVE_DIR)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Gen 0: seed
    header(f"GEN 0 -- seed ({seed_cfg.label})")
    seed_result = await evaluate_pipeline(seed, seed_cfg, eval_tag="seed", gen=0)
    print(f"  {WHITE}Gen 0:{RESET} {seed_result.win_rate} score={seed_result.composite_score:+.0f}")

    LEADERBOARD_SIZE = 5
    seed_id = f"seed_{id(seed_cfg)}"
    leaderboard: list[LeaderboardEntry] = [
        LeaderboardEntry(
            id=seed_id, label="Gen 0 (seed)", cfg=seed_cfg,
            pipeline=seed, score=seed_result.composite_score, gen=0,
        )
    ]
    broadcast_eval_result(
        f"Gen 0: {seed_cfg.label}", seed_cfg, seed_result,
        gen=0, is_best=True, in_archive=True,
    )

    score_trajectory: list[float] = [seed_result.composite_score]
    all_results: list[tuple[str, PipelineConfig, EvalResult, float]] = [
        (f"Gen 0: {seed_cfg.label}", seed_cfg, seed_result, seed_result.composite_score),
    ]

    # Evolutionary loop
    for gen in range(1, 1001):
        best_score = leaderboard[0].score if leaderboard else 0
        header(f"GEN {gen} -- top {len(leaderboard)}, best={best_score:+.0f}")

        # Step 1: Build rich context for reflection
        all_sorted = sorted(all_results, key=lambda x: x[3], reverse=True)

        def _game_summary(label, cfg, r, s):
            pipe_label = label.split(": ", 1)[-1] if ": " in label else label
            illegal = sum(len(g.get("illegal_moves", [])) for g in r.games)
            lines = [f"  {pipe_label}: score={s:+.0f} {r.win_rate} "
                     f"avg={r.avg_eval:+.0f}cp blun={r.blunder_count} "
                     f"mv={r.total_moves}"
                     + (f" ILLEGAL={illegal}" if illegal else "")]
            for g in r.games[:1]:
                moves = g.get("move_list", [])
                curve = g.get("eval_curve", [])
                move_str = " ".join(
                    f"{i // 2 + 1}.{moves[i]}"
                    + (f" {moves[i+1]}" if i + 1 < len(moves) else "")
                    for i in range(0, len(moves), 2)
                )[:150]
                lines.append(f"    Moves: {move_str}")
                if curve:
                    curve_str = ",".join(f"{c:+d}" for c in curve[-8:])
                    lines.append(f"    Eval (last 8): [{curve_str}]")
                blunders = []
                for j in range(1, len(curve)):
                    if curve[j] - curve[j-1] < -200 and j < len(moves):
                        blunders.append(
                            f"move {j//2+1} {moves[j]}: "
                            f"{curve[j-1]:+d} → {curve[j]:+d}cp"
                        )
                if blunders:
                    lines.append(f"    Blunders: {'; '.join(blunders[:3])}")
            return "\n".join(lines)

        if all_results:
            top = [_game_summary(*e) for e in all_sorted[:5]]
            bottom = [_game_summary(*e) for e in all_sorted[-3:]]
            history = (
                "BEST 5:\n" + "\n".join(top)
                + "\n\nWORST 3:\n" + "\n".join(bottom)
            )
        else:
            history = "(seed run)"

        reflection = ""

        # Opus reflection — reads game data, moves, blunders
        if all_results:
            refl_system = (
                "You are a chess coach analyzing an AI's games. "
                "Be specific about moves and positions. "
                "3-4 sentences."
            )
            refl_user = (
                f"SCORING: Sum of (cp+500)/1000 while eval > -500cp. "
                f"Draws +100K, wins +200K. Higher = better.\n\n"
                f"{history}\n\n"
                f"Best score: {best_score:+.0f}\n\n"
                f"Analyze the move sequences and eval curves. "
                f"What specific moves or patterns cause the eval "
                f"to crash? What concrete change to the prompts "
                f"or pipeline would help?"
            )
            refl_input_path = LIVE_DIR / f"_opus_reflection_gen{gen}.txt"
            refl_input_path.write_text(
                f"=== SYSTEM ===\n{refl_system}\n\n"
                f"=== USER ({len(refl_user)} chars) ===\n{refl_user}"
            )
            try:
                reflection = (await _cli_call_opus(
                    refl_system, refl_user,
                )).strip()
                if reflection:
                    print(
                        f"\n  {MAGENTA}Reflection:{RESET} "
                        f"{reflection}"
                    )
            except Exception:
                pass

        # Step 2: Generate candidates — Opus-driven
        candidates = []
        # 2a: Use leaderboard as parents — fill to CANDIDATES_PER_GEN
        lb_parents = list(leaderboard[:CANDIDATES_PER_GEN])
        while len(lb_parents) < CANDIDATES_PER_GEN:
            lb_parents.append(leaderboard[0])

        if lb_parents:
            parent_descs = []
            for i, p in enumerate(lb_parents):
                wf = p.pipeline.compile()
                # Serialize parent as the same JSON format Opus outputs
                node_specs = []
                for nid, node in wf.nodes.items():
                    if not hasattr(node, "prompt_template"):
                        continue
                    spec: dict = {"id": nid}
                    if node.prompt_template:
                        spec["prompt"] = node.prompt_template
                    if hasattr(node, "reads") and node.reads:
                        spec["reads"] = sorted(node.reads)
                    if hasattr(node, "writes") and node.writes:
                        spec["writes"] = sorted(node.writes)[0]
                    node_specs.append(spec)
                parent_json = json.dumps(
                    {"parent": i + 1, "pipeline": node_specs},
                    indent=2,
                )
                parent_descs.append(
                    f"  PARENT {i+1}: {p.label} (score={p.score:+.0f})\n"
                    f"  Current pipeline (modify this):\n{parent_json}"
                )
            opus_system = (
                "You are an optimizer for a chess AI pipeline. "
                f"Propose exactly {len(lb_parents)} experiment variants, "
                "one JSON object per line — one mutation per parent. "
                "No other text — just JSON lines."
            )
            opus_user = (
                f"SCORING: Sum of (cp+500)/1000 while above -500cp. "
                f"Draws +100K, wins +200K. Higher = better.\n\n"
                f"RESULTS:\n{history}\n\n"
                f"REFLECTION:\n{reflection or '(none)'}\n\n"
                f"PARENTS — each shows its current pipeline as JSON. "
                f"Modify the pipeline and output your variant.\n\n"
                + "\n\n".join(parent_descs) + "\n\n"
                f"INSTRUCTIONS:\n"
                f"- Output {len(lb_parents)} JSON objects, one per parent\n"
                f"- Each must have {{'parent': N, 'pipeline': [...]}}\n"
                f"- Modify the parent's pipeline: change prompts, "
                f"add nodes, remove nodes, rewire reads\n"
                f"- Nodes execute sequentially; last MUST write "
                f"to .factory/chess/move.md\n"
                f"- board_state.md and memory.md are auto-available\n"
                f"- Keep changes small — mutate 1-2 things per parent\n"
                f"- The goal: teach better chess through prompts "
                f"and architecture\n"
                f"- NOTE: A system prompt enforces a 100-word output "
                f"limit on all nodes. Write prompts that elicit "
                f"SHORT, decisive responses — not long analysis."
            )
            (LIVE_DIR / f"_opus_proposals_gen{gen}.txt").write_text(
                f"=== SYSTEM ===\n{opus_system}\n\n"
                f"=== USER ({len(opus_user)} chars) ===\n{opus_user}"
            )
            try:
                raw = await _cli_call_opus(opus_system, opus_user)
                if raw:
                    for obj in _extract_json_objects(raw):
                        pidx = obj.pop("parent", 1) - 1
                        pidx = max(0, min(pidx, len(lb_parents) - 1))
                        parent_cfg = lb_parents[pidx].cfg
                        parent_label = lb_parents[pidx].label
                        parsed = parse_opus_proposals(
                            json.dumps(obj), parent_cfg, gen,
                        )
                        if parsed:
                            label, cfg, pipeline, n_games, pmeta = parsed[0]
                            label = (
                                f"Gen {gen}.{len(candidates)+1}: "
                                f"{label.split(': ', 1)[-1]}"
                            )
                            mut_detail = {
                                "operator": "opus_guided",
                                "node": pmeta.get("node", ""),
                                **pmeta,
                            }
                            print(f"  {CYAN}> {label} [OPUS from {parent_label}]{RESET}")
                            candidates.append((label, cfg, pipeline, mut_detail))
            except Exception as exc:
                print(f"  {DIM}(opus proposals failed: {exc}){RESET}")

        if not candidates:
            print(f"  {DIM}(no candidates generated){RESET}")

        # Step 3: Evaluate candidates sequentially
        print(
            f"\n  {DIM}Evaluating {len(candidates)}"
            f" variants...{RESET}\n"
        )

        gen_candidates = []
        inserted_labels: set[str] = set()
        prev_best = best_score
        gen_start = time.monotonic()
        for label, cfg, pipeline, mut_detail in candidates:
            if mut_detail.get("after") or mut_detail.get("node_prompts"):
                mut_entry = {
                    "label": label + ":mut", "gen": gen,
                    "mutation": mut_detail, "type": "mutation_info",
                }
                with open(LIVE_DIR / "experiment_log.jsonl", "a") as f:
                    f.write(json.dumps(mut_entry) + "\n")
            result = await evaluate_pipeline(
                pipeline, cfg,
                n_games=GAMES_PER_EVAL, eval_tag=label, gen=gen,
            )
            score = result.composite_score
            new_entry = LeaderboardEntry(
                id=f"g{gen}_{len(gen_candidates)}",
                label=label, cfg=cfg, pipeline=pipeline,
                score=score, gen=gen,
                prompts=mut_detail.get("node_prompts", {}),
            )
            # Insert into leaderboard if it qualifies
            leaderboard.append(new_entry)
            leaderboard.sort(key=lambda e: e.score, reverse=True)
            inserted = new_entry in leaderboard[:LEADERBOARD_SIZE]
            leaderboard = leaderboard[:LEADERBOARD_SIZE]
            gen_candidates.append((label, cfg, result, score))
            if inserted:
                inserted_labels.add(label)
            marker = f" {GREEN}-> top {LEADERBOARD_SIZE}{RESET}" if inserted else ""
            print(
                f"  {WHITE}{label}:{RESET} {result.win_rate}"
                f" score={score:+.0f}"
                f" (avg={result.avg_eval:+.0f}cp"
                f" blun={result.blunder_count}"
                f" mv={result.total_moves}){marker}"
            )
            broadcast_eval_result(
                label, cfg, result, gen=gen,
                is_best=False,
                in_archive=(label in inserted_labels),
                mutation_detail=mut_detail,
            )

        new_score = leaderboard[0].score if leaderboard else 0
        score_trajectory.append(new_score)
        if new_score > prev_best:
            print(
                f"\n  {GREEN}NEW BEST: score={new_score:+.0f}{RESET}"
            )

        # Broadcast leaderboard
        archive_entries = [
            {"id": e.id[:8], "score": e.score, "gen": e.gen,
             "config": e.label.split(": ", 1)[-1] if ": " in e.label else e.label,
             "features": []}
            for e in leaderboard
        ]
        broadcast_archive(archive_entries)
        if reflection:
            entry = {"type": "reflection", "gen": gen, "text": reflection}
            with open(LIVE_DIR / "experiment_log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            with open(LIVE_DIR / "recording.jsonl", "a") as f:
                f.write(json.dumps({"t": time.monotonic(), **entry}) + "\n")

        all_results.extend(gen_candidates)
        import resource
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
        print(
            f"  {DIM}Gen {gen} in {time.monotonic() - gen_start:.0f}s"
            f" | mem={mb}MB | top={len(leaderboard)}{RESET}"
        )

    header("RESULTS")
    best = leaderboard[0] if leaderboard else None
    if best:
        print(f"\n  {GREEN}Winner:{RESET} score={best.score:+.0f}")
        print(f"  {GREEN}Pipeline:{RESET} {best.label}")
