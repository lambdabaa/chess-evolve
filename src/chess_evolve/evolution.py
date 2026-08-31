"""Outer loop: evolutionary prompt optimization using factory's MAP-Elites."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import shutil
import time

from factory.cycle_analyzer import CycleRecord
from factory.outer_loop.models import MutationType
from factory.outer_loop.mutations import WeightedRandomStrategy, apply_random_mutation
from factory.outer_loop.population import MAPElitesArchive, Population
from factory.outer_loop.reflector import OuterLoopReflector

from chess_evolve.broadcast import broadcast_archive, broadcast_eval_result
from chess_evolve.config import CANDIDATES_PER_GEN, GAMES_PER_EVAL, LIVE_DIR
from chess_evolve.display import CYAN, DIM, GREEN, MAGENTA, RESET, WHITE, YELLOW, header, print
from chess_evolve.engine import _cli_call_opus
from chess_evolve.game import EvalResult, evaluate_pipeline
from chess_evolve.pipeline import KNOB_SPACE, PipelineConfig, build_pipeline
from chess_evolve.prompts import PROMPT_REGISTRY, _register_prompt


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


def parse_opus_proposals(
    raw: str, best_cfg: PipelineConfig, gen: int,
) -> list[tuple[str, PipelineConfig, object, int]]:
    """Parse Opus JSON proposals into (label, cfg, pipeline, n_games) tuples."""
    candidates: list[tuple[str, PipelineConfig, object, int]] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            prop = json.loads(line)
            n_games = prop.get("games", GAMES_PER_EVAL)
            if "knobs" in prop and isinstance(prop["knobs"], dict):
                overrides = {k: v for k, v in prop["knobs"].items() if hasattr(best_cfg, k)}
                for k, v in overrides.items():
                    if "prompt" in prop:
                        _register_prompt(k, v, prop["prompt"])
                    for _, (kn, choices) in enumerate(KNOB_SPACE):
                        if kn == k and v not in choices:
                            choices.append(v)
                child_cfg = dataclasses.replace(best_cfg, **overrides)
                desc = " + ".join(f"{k}={v}" for k, v in overrides.items())
            elif prop.get("rerun"):
                child_cfg = dataclasses.replace(best_cfg)
                desc = "rerun (variance check)"
            else:
                knob, value = prop["knob"], prop["value"]
                if not hasattr(best_cfg, knob):
                    continue
                if "prompt" in prop and isinstance(prop["prompt"], str):
                    _register_prompt(knob, value, prop["prompt"])
                    print(f"  {CYAN}NEW PROMPT: {knob}={value}{RESET}")
                for _, (kn, choices) in enumerate(KNOB_SPACE):
                    if kn == knob and value not in choices:
                        choices.append(value)
                child_cfg = dataclasses.replace(best_cfg, **{knob: value})
                desc = f"{knob}={value}"
            candidates.append((
                f"Gen {gen}.{len(candidates)+1}: {desc}",
                child_cfg, build_pipeline(child_cfg), n_games,
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

    archive = MAPElitesArchive()
    seed_ind = Population.make_individual(
        seed.compile(), generation=0, score=seed_result.composite_score
    )
    archive.add(seed_ind)
    ind_configs: dict[str, PipelineConfig] = {seed_ind.id: seed_cfg}
    ind_prompts: dict[str, dict[str, str]] = {}
    ind_labels: dict[str, str] = {seed_ind.id: "Gen 0 (seed)"}
    broadcast_eval_result(
        f"Gen 0: {seed_cfg.label}", seed_cfg, seed_result,
        gen=0, is_best=True, in_archive=True,
    )

    reflector = OuterLoopReflector(k=3)
    score_trajectory: list[float] = [seed_result.composite_score]
    cycle_records: dict[str, CycleRecord] = {seed_ind.id: seed_result.to_cycle_record(0)}
    PLATEAU_WINDOW, PLATEAU_THRESHOLD = 10, 5.0
    all_results: list[tuple[str, PipelineConfig, EvalResult, float]] = [
        (f"Gen 0: {seed_cfg.label}", seed_cfg, seed_result, seed_result.composite_score),
    ]

    # Evolutionary loop
    for gen in range(1, 1001):
        best_ind = archive.best()
        best_score = best_ind.score if best_ind else 0
        header(f"GEN {gen} -- archive: {archive.size} cells, best={best_score:+.0f}")

        stalled = (len(score_trajectory) > PLATEAU_WINDOW and
                   all(abs(s - score_trajectory[-(PLATEAU_WINDOW+1)]) < PLATEAU_THRESHOLD
                       for s in score_trajectory[-PLATEAU_WINDOW:]))

        # Step 1: Contrastive reflection
        all_sorted = sorted(all_results, key=lambda x: x[3], reverse=True)
        def _fmt(label, cfg, r, s):
            return (
                f"  {cfg.label}: {r.win_rate} "
                f"avg={r.avg_eval:+.0f}cp "
                f"blun={r.blunder_count} mv={r.total_moves} "
                f"score={s:+.0f}"
            )
        if all_results:
            top = [_fmt(*e) for e in all_sorted[:10]]
            bottom = [_fmt(*e) for e in all_sorted[-10:]]
            history = (
                "TOP 10 (best):\n" + "\n".join(top)
                + "\n\nBOTTOM 10 (worst):\n" + "\n".join(bottom)
            )
        else:
            history = "(seed run)"

        reflection_report = None
        reflection = ""
        if cycle_records:
            record_ids = list(cycle_records.keys())
            records = [(iid, cycle_records[iid].score_end or 0, cycle_records[iid])
                       for iid in record_ids]
            knob_vals = {}
            for iid in record_ids:
                cfg = ind_configs.get(iid)
                if cfg:
                    vals = {k: getattr(cfg, k) for k, _ in KNOB_SPACE if hasattr(cfg, k)}
                    prompts = ind_prompts.get(iid, {})
                    for node, prompt in prompts.items():
                        vals[f"prompt_{node}"] = prompt[:100]
                    knob_vals[iid] = vals
            # Factory's structural reflector (for guided mutations)
            try:
                reflection_report = reflector.reflect(
                    records, gen, knob_values_by_id=knob_vals,
                )
                if reflection_report.mutation_suggestions:
                    sug = '; '.join(
                        reflection_report.mutation_suggestions[:3],
                    )
                    for iid, lbl in ind_labels.items():
                        sug = sug.replace(iid[:8], lbl)
                    print(
                        f"\n  {DIM}Knob gradients: {sug}{RESET}"
                    )
            except Exception as exc:
                print(f"  {DIM}(reflector error: {exc}){RESET}")

        # Opus reflection (primary — reads game data)
        if all_results:
            try:
                reflection = (await _cli_call_opus(
                    "You are a chess coach analyzing an AI's games. "
                    "Be specific about moves and positions. "
                    "3-4 sentences.",
                    f"SCORING: Each variant is scored by the number "
                    f"of moves where eval stays above -500cp "
                    f"(game ends when it dips below). "
                    f"Draws add 100, wins add 200.\n\n"
                    f"{history}\n\n"
                    f"Archive: {archive.size} cells, "
                    f"best={best_score:+.0f}\n\n"
                    f"What causes the eval to crash below "
                    f"-500cp? What concrete change would help "
                    f"the AI survive longer?",
                )).strip()
                if reflection:
                    for iid, lbl in ind_labels.items():
                        reflection = reflection.replace(
                            iid[:8], lbl,
                        )
                    print(
                        f"\n  {MAGENTA}Reflection:{RESET} "
                        f"{reflection}"
                    )
            except Exception:
                pass

        # Game-aware prompt hints from blunder analysis
        if reflection_report and all_results:
            worst = sorted(all_results, key=lambda x: x[3])[:3]
            blunder_details = []
            for _, cfg, result, score in worst:
                for g in result.games:
                    curve = g.get("eval_curve", [])
                    moves = g.get("move_list", [])
                    for i in range(1, len(curve)):
                        if curve[i] - curve[i-1] < -200 and i < len(moves):
                            blunder_details.append(
                                f"Move {i}: {moves[i]} dropped {curve[i-1]:+d}cp to {curve[i]:+d}cp"
                            )
            if blunder_details:
                try:
                    coach_prompt = (
                        "You are a chess coach. Write ONE sentence"
                        " of advice to prevent these blunders. "
                        "Be specific about the pattern."
                    )
                    hint = (await _cli_call_opus(
                        coach_prompt,
                        "Recent blunders:\n" + "\n".join(blunder_details[:8]),
                    )).strip()
                    if hint and reflection_report:
                        reflection_report.prompt_improvements.append(hint)
                        print(f"  {CYAN}Prompt hint:{RESET} {hint}")
                except Exception:
                    pass

        # Step 2: Mutation
        strategy = WeightedRandomStrategy(weights={
            MutationType.NODE_INSERT.value: 0,
            MutationType.NODE_REMOVE.value: 0,
            MutationType.EDGE_REDIRECT.value: 0,
            MutationType.PARALLELIZE.value: 0,
            MutationType.SERIALIZE.value: 0,
            MutationType.PARAM_MUTATE.value: 0,
            MutationType.PROMPT_MUTATE.value: 0.50,
            MutationType.KNOB_MUTATE.value: 0.50,
        })
        if stalled and hasattr(strategy, "on_plateau"):
            strategy.on_plateau()

        # Step 2b: Generate candidates sequentially (prompt rewrites are sync)
        candidates = []
        for c in range(CANDIDATES_PER_GEN):
            parent = archive.sample_parent(
                tournament_size=5, rank_weighted=True,
            )
            if parent is None:
                continue
            parent_cfg = ind_configs.get(parent.id, seed_cfg)
            parent_wf = build_pipeline(parent_cfg).compile()

            mut = apply_random_mutation(
                parent_wf, strategy, gen,
                reflection_report=reflection_report,
            )
            if mut is None:
                continue
            child_wf, rec = mut

            def _coerce_back(k, v):
                orig = getattr(parent_cfg, k, None)
                if isinstance(orig, bool):
                    return (
                        v == "True" if isinstance(v, str)
                        else bool(v)
                    )
                if isinstance(orig, int):
                    return int(v) if not isinstance(v, int) else v
                return v

            overrides = {
                k: _coerce_back(k, v)
                for k, v in child_wf.knob_values.items()
                if hasattr(parent_cfg, k)
                and getattr(parent_cfg, k) != _coerce_back(k, v)
            }
            if overrides:
                child_cfg = dataclasses.replace(parent_cfg, **overrides)
                desc = " + ".join(
                    f"{k}={v}" for k, v in overrides.items()
                )
            else:
                child_cfg = parent_cfg
                desc = rec.rationale or "no-op"
            child_pipeline = build_pipeline(child_cfg)
            for k, v in child_wf.knob_values.items():
                if k.startswith("_prompt_"):
                    child_pipeline.graph.knob_values[k] = v
                    child_pipeline.graph.knob_expandable[k] = (
                        child_wf.knob_expandable.get(k, "")
                    )
            label = f"Gen {gen}.{c+1}: {desc}"
            prompt_detail: dict[str, str] = {}
            if rec.operator.value == "prompt_mutate" and rec.target_node:
                nid = rec.target_node
                new_prompt = child_wf.knob_values.get(
                    f"_prompt_{nid}", "",
                )
                old_node = (
                    build_pipeline(parent_cfg).graph.nodes.get(nid)
                )
                old_prompt = ""
                if old_node and hasattr(old_node, "prompt_template"):
                    old_prompt = old_node.prompt_template or ""
                prompt_detail = {
                    "node": nid,
                    "before": old_prompt,
                    "after": str(new_prompt),
                }
            mut_detail = {
                "operator": rec.operator.value,
                "node": rec.target_node or "",
                **prompt_detail,
            }
            print(f"  {YELLOW}> {label}{RESET}")
            candidates.append(
                (label, child_cfg, child_pipeline, mut_detail)
            )

        # Step 3: Evaluate all candidates in parallel
        print(
            f"\n  {DIM}Evaluating {len(candidates)}"
            f" variants...{RESET}\n"
        )

        async def eval_one(label, cfg, pipeline, mut_detail):
            # Write mutation detail to log so UI tooltip works during play
            if mut_detail.get("after"):
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
            return label, cfg, pipeline, result, mut_detail

        gen_candidates = []
        inserted_labels: set[str] = set()
        prev_best = best_score
        gen_start = time.monotonic()
        tasks = [
            asyncio.create_task(eval_one(*c))
            for c in candidates
        ]
        for coro in asyncio.as_completed(tasks):
            entry = await coro
            label, cfg, pipeline, result, mut_detail = entry
            score = result.composite_score
            ind = Population.make_individual(
                pipeline.compile(), generation=gen, score=score,
            )
            inserted = archive.add(ind)
            ind_configs[ind.id] = cfg
            ind_labels[ind.id] = label
            cycle_records[ind.id] = result.to_cycle_record(gen)
            if mut_detail.get("after"):
                ind_prompts[ind.id] = {
                    mut_detail["node"]: mut_detail["after"],
                }
            gen_candidates.append((label, cfg, result, score))
            if inserted:
                inserted_labels.add(label)
            marker = f" {GREEN}-> archive{RESET}" if inserted else ""
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

        new_score = archive.best().score if archive.best() else 0
        score_trajectory.append(new_score)
        if new_score > prev_best:
            print(
                f"\n  {GREEN}NEW BEST: score={new_score:+.0f}"
                f" (archive: {archive.size} cells){RESET}"
            )

        # Broadcast current archive population
        archive_entries = [
            {"id": ind.id[:8], "score": ind.score, "gen": ind.generation,
             "config": ind_configs.get(ind.id, seed_cfg).label,
             "features": list(ind.features)}
            for ind in archive.all_individuals()
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
            f" | mem={mb}MB | archive={archive.size}{RESET}"
        )

    header("RESULTS")
    best = archive.best()
    print(f"\n  {GREEN}Winner:{RESET} score={best.score:+.0f}")
    print(f"  {GREEN}Archive:{RESET} {archive.size} cells")
    print(f"  {GREEN}Config:{RESET} {ind_configs[best.id].full_label}")
