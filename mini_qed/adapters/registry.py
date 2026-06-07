"""Adapter factory: build adapters from config dicts."""

import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.adapters.openai_compatible import OpenAICompatibleAdapter
from mini_qed.adapters.anthropic_adapter import AnthropicAdapter
from mini_qed.adapters.openai_adapter import OpenAIAdapter


def build_adapter(name: str, cfg: dict) -> AbstractLLMAdapter:
    """Create a single adapter from a config sub-dict.

    Args:
        name: Adapter name (for error messages).
        cfg: Dict with keys: type, base_url, api_key, model (optional).

    Returns:
        A concrete AbstractLLMAdapter instance.
    """
    adapter_type = cfg["type"]

    if adapter_type == "openai_compatible":
        return OpenAICompatibleAdapter(
            base_url=cfg["base_url"],
            api_key=os.path.expandvars(cfg["api_key"]),
            model=cfg.get("model", ""),
        )
    elif adapter_type == "anthropic_sdk":
        return AnthropicAdapter(
            api_key=os.path.expandvars(cfg["api_key"]),
            model=cfg.get("model", "claude-sonnet-4-6"),
        )
    elif adapter_type == "openai_sdk":
        return OpenAIAdapter(
            api_key=os.path.expandvars(cfg["api_key"]),
            model=cfg.get("model", "gpt-5.1"),
        )
    else:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}' for adapter '{name}'. "
            f"Supported types: openai_compatible, anthropic_sdk, openai_sdk."
        )


class AdapterRegistry:
    """Holds adapter configs and creates adapter instances on demand.

    Adapters are cached after first creation so each adapter is a singleton
    within a pipeline run (the underlying AsyncOpenAI client is reused).

    Usage:
        reg = AdapterRegistry({"adapters": {"deepseek": {...}}})
        adapter = reg.get("deepseek", model="deepseek-v4-pro")
    """

    def __init__(self, config: dict):
        self._configs: dict[str, dict] = config.get("adapters", {})
        self._instances: dict[str, AbstractLLMAdapter] = {}

    def get(self, name: str, *, model: str | None = None) -> AbstractLLMAdapter:
        """Get or create an adapter instance.

        Args:
            name: Adapter name (must exist in config.adapters).
            model: Override the default model for this adapter.

        Returns:
            A cached or newly-created AbstractLLMAdapter.
        """
        cfg = self._configs.get(name)
        if cfg is None:
            raise ValueError(
                f"Adapter '{name}' not found in config.adapters. "
                f"Available: {', '.join(self._configs.keys())}"
            )

        # If model override requested, always create a new instance
        if model is not None:
            cfg_copy = dict(cfg)
            cfg_copy["model"] = model
            return build_adapter(name, cfg_copy)

        # Otherwise cache and reuse
        if name not in self._instances:
            self._instances[name] = build_adapter(name, cfg)
        return self._instances[name]
