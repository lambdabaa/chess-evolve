"""Tests for pipeline definition and factory Package integration."""

from __future__ import annotations

from factory.workflow.package import Package
from factory.workflow.primitives import AgentNode, Workflow

from chess_evolve.pipeline import KNOB_SPACE, PipelineConfig, _PROMPT_NODES, build_pipeline


class TestPipelineConfig:
    def test_defaults_are_valid(self):
        cfg = PipelineConfig()
        assert cfg.model == "haiku"
        assert cfg.pipeline_mode == "parallel"
        assert cfg.verify_iterations == 2
        assert cfg.verify_iterations == 2

    def test_label_includes_mode(self, default_config):
        assert default_config.label == "seed"

    def test_full_label_includes_knobs(self, default_config):
        label = default_config.full_label
        for knob_name, _ in KNOB_SPACE:
            assert knob_name in label


class TestBuildPipeline:
    def test_returns_package(self):
        pipeline = build_pipeline()
        assert isinstance(pipeline, Package)

    def test_has_entry_and_exit_nodes(self):
        pipeline = build_pipeline()
        assert pipeline.entry_node
        assert pipeline.exit_node

    def test_compile_produces_workflow(self):
        pipeline = build_pipeline()
        wf = pipeline.compile()
        assert isinstance(wf, Workflow)

    def test_compiled_node_count(self):
        pipeline = build_pipeline()
        wf = pipeline.compile()
        assert len(wf.nodes) >= 5  # at least analyst, tact, pos, selector, verifier

    def test_compiled_has_knob_values(self):
        pipeline = build_pipeline()
        wf = pipeline.compile()
        assert wf.knob_values
        for knob_name, _ in KNOB_SPACE:
            assert knob_name in wf.knob_values

    def test_compiled_has_agent_nodes(self):
        pipeline = build_pipeline()
        wf = pipeline.compile()
        agent_ids = [nid for nid, n in wf.nodes.items() if isinstance(n, AgentNode)]
        assert len(agent_ids) >= 4

    def test_prompt_nodes_exist_in_compiled(self):
        pipeline = build_pipeline()
        wf = pipeline.compile()
        for nid in _PROMPT_NODES:
            assert nid in wf.nodes, f"Expected node '{nid}' in compiled workflow"

    def test_without_verification(self):
        cfg = PipelineConfig(verify_iterations=0)
        pipeline = build_pipeline(cfg)
        wf = pipeline.compile()
        assert "verify_gate" not in wf.nodes

    def test_with_phase_routing(self):
        cfg = PipelineConfig(game_phase_routing=True)
        pipeline = build_pipeline(cfg)
        wf = pipeline.compile()
        assert "phase_gate" in wf.nodes


class TestKnobSpaceConsistency:
    def test_knob_names_match_config_fields(self):
        cfg = PipelineConfig()
        for knob_name, _ in KNOB_SPACE:
            assert hasattr(cfg, knob_name), f"KNOB_SPACE entry '{knob_name}' not in PipelineConfig"

    def test_knob_defaults_in_bounds(self):
        cfg = PipelineConfig()
        for knob_name, choices in KNOB_SPACE:
            val = getattr(cfg, knob_name)
            assert val in choices, f"Default {knob_name}={val} not in bounds {choices}"


class TestPromptKnobRoundTrip:
    def test_prompt_knob_stored_on_graph(self):
        pipeline = build_pipeline()
        pipeline.graph.knob_values["_prompt_tactician"] = "Rewritten tactician prompt"
        pipeline.graph.knob_expandable["_prompt_tactician"] = "Prompt for tactician"
        assert pipeline.graph.knob_values["_prompt_tactician"] == "Rewritten tactician prompt"

    def test_prompt_knob_absent_uses_default(self):
        pipeline = build_pipeline()
        wf = pipeline.compile()
        node = wf.nodes["tactician"]
        assert isinstance(node, AgentNode)
        assert node.prompt_template
        assert "_prompt_" not in node.prompt_template
