"""Configuration loading, validation, and env-var expansion."""

from dataclasses import dataclass, field
import os
import re


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} patterns in a string using environment variables."""
    def _replacer(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            raise ValueError(
                f"Environment variable '{var}' is not set. "
                f"Required by config value that references ${{{var}}}."
            )
        return val
    return _ENV_VAR_RE.sub(_replacer, value)


@dataclass
class AgentRoleConfig:
    """Configuration for a single agent role (proof search, verifier, etc.)."""
    adapter: str
    model: str
    thinking: bool = False
    reasoning_effort: str | None = None

    @classmethod
    def from_dict(cls, d: dict, name: str) -> "AgentRoleConfig":
        adapter = d.get("adapter")
        if not adapter:
            raise ValueError(
                f"Agent '{name}': missing required field 'adapter'."
            )
        return cls(
            adapter=adapter,
            model=d.get("model", ""),
            thinking=d.get("thinking", False),
            reasoning_effort=d.get("reasoning_effort"),
        )


@dataclass
class SimpleModeConfig:
    """Simple mode pipeline configuration."""
    proof_search: AgentRoleConfig
    structural_verifier: AgentRoleConfig
    detailed_verifier: AgentRoleConfig
    verdict: AgentRoleConfig
    brainstorm: dict | None = None          # {enabled: bool, providers: [...]}
    multi_model: dict | None = None         # {enabled: bool, ...}

    @classmethod
    def from_dict(cls, d: dict) -> "SimpleModeConfig":
        required = {
            "proof_search": "simple_mode.proof_search",
            "structural_verifier": "simple_mode.structural_verifier",
            "detailed_verifier": "simple_mode.detailed_verifier",
            "verdict": "simple_mode.verdict",
        }
        for key, path in required.items():
            if key not in d:
                raise ValueError(
                    f"Missing required field '{path}' in config.yaml pipeline.simple_mode."
                )

        return cls(
            proof_search=AgentRoleConfig.from_dict(d["proof_search"], "simple_mode.proof_search"),
            structural_verifier=AgentRoleConfig.from_dict(d["structural_verifier"], "simple_mode.structural_verifier"),
            detailed_verifier=AgentRoleConfig.from_dict(d["detailed_verifier"], "simple_mode.detailed_verifier"),
            verdict=AgentRoleConfig.from_dict(d["verdict"], "simple_mode.verdict"),
            brainstorm=d.get("brainstorm"),
            multi_model=d.get("multi_model"),
        )


@dataclass
class PipelineConfig:
    """Full pipeline configuration, parsed from config.yaml."""
    max_proof_iterations: int
    output_dir: str
    literature_survey: AgentRoleConfig
    simple_mode: SimpleModeConfig
    proof_summary: AgentRoleConfig

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "PipelineConfig":
        """Parse the top-level config dict (already YAML-loaded).

        Validates all required sections and fields.
        """
        pipeline = raw.get("pipeline")
        if not pipeline:
            raise ValueError(
                "Missing top-level 'pipeline' section in config.yaml."
            )

        # Required sub-sections
        required_sections = {
            "literature_survey": "pipeline.literature_survey",
            "simple_mode": "pipeline.simple_mode",
            "proof_summary": "pipeline.proof_summary",
        }
        for key, path in required_sections.items():
            if key not in pipeline:
                raise ValueError(f"Missing required section '{path}' in config.yaml.")

        return cls(
            max_proof_iterations=pipeline.get("max_proof_iterations", 9),
            output_dir=pipeline.get("output_dir", "./proof_output"),
            literature_survey=AgentRoleConfig.from_dict(
                pipeline["literature_survey"], "pipeline.literature_survey"
            ),
            simple_mode=SimpleModeConfig.from_dict(pipeline["simple_mode"]),
            proof_summary=AgentRoleConfig.from_dict(
                pipeline["proof_summary"], "pipeline.proof_summary"
            ),
        )


def load_pipeline_config(path: str) -> tuple[PipelineConfig, dict]:
    """Load and validate pipeline config from a YAML file.

    Returns:
        (PipelineConfig, raw_config_dict) — the raw dict is for AdapterRegistry.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Expand env vars in the entire config tree
    raw = _expand_config_env_vars(raw)

    pipeline = PipelineConfig.from_yaml_dict(raw)
    return pipeline, raw


def _expand_config_env_vars(obj):
    """Recursively expand ${VAR} patterns in all string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    elif isinstance(obj, dict):
        return {k: _expand_config_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_config_env_vars(v) for v in obj]
    return obj
