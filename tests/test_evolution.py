"""Tests for the evolution loop utilities and factory integration."""

from __future__ import annotations

import random

from factory.outer_loop.similarity import compute_features
from factory.workflow.primitives import AgentNode

from chess_evolve.evolution import mutate_knobs
from chess_evolve.pipeline import KNOB_SPACE, PipelineConfig, build_pipeline


class TestFactoryFeatures:
    """Verify factory's compute_features differentiates our knob combos."""

    def test_different_knobs_different_features(self):
        wf1 = build_pipeline(PipelineConfig(verify_style="strict")).compile()
        wf2 = build_pipeline(PipelineConfig(verify_style="lenient")).compile()
        assert compute_features(wf1) != compute_features(wf2)

    def test_different_prompts_different_features(self):
        cfg = PipelineConfig()
        wf1 = build_pipeline(cfg).compile()
        wf2 = build_pipeline(cfg).compile()
        node = wf2.nodes["tactician"]
        assert isinstance(node, AgentNode)
        wf2.nodes["tactician"] = node.model_copy(
            update={"prompt_template": "completely different prompt"}
        )
        assert compute_features(wf1) != compute_features(wf2)

    def test_same_config_same_features(self):
        cfg = PipelineConfig()
        wf = build_pipeline(cfg).compile()
        assert compute_features(wf) == compute_features(wf)

    def test_fixed_length(self):
        wf1 = build_pipeline(PipelineConfig(verify_iterations=2)).compile()
        wf2 = build_pipeline(PipelineConfig(verify_iterations=0)).compile()
        assert len(compute_features(wf1)) == len(compute_features(wf2))

    def test_all_knob_combos_unique(self):
        import dataclasses
        seen = set()
        for vs in ["strict", "lenient"]:
            for vi in [0, 2]:
                for oh in ["theory", "principled"]:
                    cfg = dataclasses.replace(
                        PipelineConfig(),
                        verify_style=vs,
                        verify_iterations=vi,
                        opening_hint=oh,
                    )
                    f = compute_features(build_pipeline(cfg).compile())
                    seen.add(f)
        assert len(seen) == 2 * 2 * 2


class TestMutateKnobs:
    def test_changes_exactly_one_field(self, default_config):
        rng = random.Random(42)
        new_cfg, desc = mutate_knobs(default_config, rng)
        diffs = 0
        for knob_name, _ in KNOB_SPACE:
            if getattr(new_cfg, knob_name) != getattr(default_config, knob_name):
                diffs += 1
        assert diffs == 1 or desc == "no-op"

    def test_returns_description(self, default_config):
        rng = random.Random(42)
        _, desc = mutate_knobs(default_config, rng)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_new_value_in_bounds(self, default_config):
        rng = random.Random(42)
        new_cfg, desc = mutate_knobs(default_config, rng)
        if desc != "no-op":
            knob_name = desc.split("=")[0]
            new_val = getattr(new_cfg, knob_name)
            choices = [
                c for _, (name, c) in enumerate(KNOB_SPACE) if name == knob_name
            ][0]
            assert new_val in choices

    def test_deterministic_with_seed(self, default_config):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        cfg1, desc1 = mutate_knobs(default_config, rng1)
        cfg2, desc2 = mutate_knobs(default_config, rng2)
        assert desc1 == desc2
