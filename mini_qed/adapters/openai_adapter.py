"""OpenAI adapter -- placeholder for Phase 2 activation."""

from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse


class OpenAIAdapter(AbstractLLMAdapter):
    """Adapter for OpenAI GPT via the official openai SDK.

    Phase 1: placeholder -- raises NotImplementedError.
    Phase 2: full implementation with openai SDK.
    """

    def __init__(self, api_key: str, model: str = "gpt-5.1"):
        self._api_key = api_key
        self._model = model

    async def chat(self, prompt: str, *, system: str = "", temperature: float = 0.7,
                   max_tokens: int = 65536, thinking: bool = False,
                   reasoning_effort: str | None = None) -> LLMResponse:
        raise NotImplementedError(
            "OpenAIAdapter is a Phase 2 placeholder. "
            "Use deepseek or opencode_go adapter for Phase 1."
        )
