"""Tests for the evolution loop utilities and factory integration."""

from __future__ import annotations

import random

from factory.workflow.primitives import AgentNode, AgentRole, Workflow

from chess_evolve.evolution import chess_features, mutate_knobs
from chess_evolve.pipeline import KNOB_SPACE, PipelineConfig, _PROMPT_NODES, build_pipeline


class TestChessFeatures:
    def test_returns_tuple(self, default_config):
        features = chess_features(default_config)
        assert isinstance(features, tuple)

    def test_knob_only_dims(self, default_config):
        features = chess_features(default_config, wf=None)
        assert len(features) == 5  # 5 knob dimensions

    def test_with_workflow_adds_prompt_dims(self, default_config):
        wf = build_pipeline(default_config).compile()
        features = chess_features(default_config, wf)
        assert len(features) == 5 + len(_PROMPT_NODES)

    def test_different_prompts_different_features(self, default_config):
        wf1 = build_pipeline(default_config).compile()
        wf2 = build_pipeline(default_config).compile()
        node = wf2.nodes["tactician"]
        assert isinstance(node, AgentNode)
        wf2.nodes["tactician"] = node.model_copy(
            update={"prompt_template": "completely different prompt text for testing"}
        )
        f1 = chess_features(default_config, wf1)
        f2 = chess_features(default_config, wf2)
        assert f1 != f2

    def test_same_config_same_features(self, default_config):
        wf = build_pipeline(default_config).compile()
        f1 = chess_features(default_config, wf)
        f2 = chess_features(default_config, wf)
        assert f1 == f2

    def test_different_knobs_different_features(self):
        cfg1 = PipelineConfig(use_verification=True)
        cfg2 = PipelineConfig(use_verification=False)
        f1 = chess_features(cfg1)
        f2 = chess_features(cfg2)
        assert f1 != f2


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
            choices = [c for _, (name, c) in enumerate(KNOB_SPACE) if name == knob_name][0]
            assert new_val in choices

    def test_deterministic_with_seed(self, default_config):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        cfg1, desc1 = mutate_knobs(default_config, rng1)
        cfg2, desc2 = mutate_knobs(default_config, rng2)
        assert desc1 == desc2
