"""Tests for config loading and validation."""

import os
import tempfile
import pytest
from mini_qed.config import (
    AgentRoleConfig,
    SimpleModeConfig,
    PipelineConfig,
    load_pipeline_config,
)


class TestAgentRoleConfig:
    def test_from_yaml_dict_basic(self):
        d = {"adapter": "deepseek", "model": "deepseek-v4-pro"}
        cfg = AgentRoleConfig.from_dict(d, "test_role")
        assert cfg.adapter == "deepseek"
        assert cfg.model == "deepseek-v4-pro"
        assert cfg.thinking is False
        assert cfg.reasoning_effort is None

    def test_from_yaml_dict_with_thinking(self):
        d = {"adapter": "deepseek", "model": "deepseek-v4-pro",
             "thinking": True, "reasoning_effort": "max"}
        cfg = AgentRoleConfig.from_dict(d, "test_role")
        assert cfg.thinking is True
        assert cfg.reasoning_effort == "max"


class TestPipelineConfig:
    def test_valid_config_loads(self):
        cfg = PipelineConfig.from_yaml_dict({
            "pipeline": {
                "max_proof_iterations": 9,
                "output_dir": "./proof_output",
                "literature_survey": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
                "simple_mode": {
                    "proof_search": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
                    "structural_verifier": {"adapter": "deepseek", "model": "deepseek-v4-flash"},
                    "detailed_verifier": {"adapter": "deepseek", "model": "deepseek-v4-flash"},
                    "verdict": {"adapter": "deepseek", "model": "deepseek-v4-flash"},
                },
                "proof_summary": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
            }
        })
        assert cfg.max_proof_iterations == 9
        assert cfg.simple_mode.proof_search.adapter == "deepseek"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match="Missing required section"):
            PipelineConfig.from_yaml_dict({
                "pipeline": {
                    "literature_survey": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
                    "proof_summary": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
                    # missing simple_mode
                }
            })

    def test_env_var_expansion(self):
        os.environ["TEST_API_KEY"] = "sk-test-123"
        cfg = PipelineConfig.from_yaml_dict({
            "pipeline": {
                "max_proof_iterations": 5,
                "literature_survey": {"adapter": "test", "model": "m1"},
                "simple_mode": {
                    "proof_search": {"adapter": "test", "model": "m1"},
                    "structural_verifier": {"adapter": "test", "model": "m2"},
                    "detailed_verifier": {"adapter": "test", "model": "m2"},
                    "verdict": {"adapter": "test", "model": "m2"},
                },
                "proof_summary": {"adapter": "test", "model": "m1"},
            }
        })
        assert cfg.max_proof_iterations == 5
        del os.environ["TEST_API_KEY"]


class TestLoadPipelineConfig:
    def test_loads_from_yaml_file(self):
        """Load pipeline config from a temporary YAML file."""
        yaml_content = """
pipeline:
  max_proof_iterations: 3
  output_dir: ./test_output
  literature_survey:
    adapter: deepseek
    model: deepseek-v4-pro
  simple_mode:
    proof_search:
      adapter: deepseek
      model: deepseek-v4-pro
    structural_verifier:
      adapter: deepseek
      model: deepseek-v4-flash
    detailed_verifier:
      adapter: deepseek
      model: deepseek-v4-flash
    verdict:
      adapter: deepseek
      model: deepseek-v4-flash
  proof_summary:
    adapter: deepseek
    model: deepseek-v4-pro
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            pipeline_cfg, raw_dict = load_pipeline_config(tmp_path)
            assert pipeline_cfg.max_proof_iterations == 3
            assert pipeline_cfg.output_dir == "./test_output"
            assert pipeline_cfg.literature_survey.adapter == "deepseek"
            assert pipeline_cfg.simple_mode.proof_search.model == "deepseek-v4-pro"
            assert pipeline_cfg.simple_mode.structural_verifier.model == "deepseek-v4-flash"
            assert pipeline_cfg.simple_mode.detailed_verifier.model == "deepseek-v4-flash"
            assert pipeline_cfg.simple_mode.verdict.model == "deepseek-v4-flash"
            assert pipeline_cfg.proof_summary.adapter == "deepseek"
            assert isinstance(raw_dict, dict)
        finally:
            os.unlink(tmp_path)
