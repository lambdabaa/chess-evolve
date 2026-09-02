"""Tests for the evolution loop utilities."""

from __future__ import annotations

import random

from chess_evolve.evolution import mutate_knobs
from chess_evolve.pipeline import KNOB_SPACE, PipelineConfig, build_pipeline


class TestBuildAndCompile:
    def test_default_pipeline_compiles(self):
        wf = build_pipeline().compile()
        assert "generator" in wf.nodes
        assert len(wf.nodes) >= 2

    def test_different_configs_produce_different_workflows(self):
        wf1 = build_pipeline(PipelineConfig(max_retries=1)).compile()
        wf2 = build_pipeline(PipelineConfig(max_retries=5)).compile()
        assert wf1.knob_values != wf2.knob_values


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
