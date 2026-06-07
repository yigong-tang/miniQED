"""Abstract interface for all LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    elapsed_s: float


class AbstractLLMAdapter(ABC):
    """All LLM backends implement this interface.

    Each concrete adapter handles one provider (DeepSeek, GLM, Claude, etc.).
    The pipeline never knows which provider it's talking to.
    """

    @abstractmethod
    async def chat(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 65536,
        thinking: bool = False,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a prompt and return a unified response.

        Args:
            prompt: The user message / task description.
            system: Optional system prompt (sent in the dedicated system role, not concatenated).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            thinking: If True, enable extended thinking (DeepSeek, Claude).
            reasoning_effort: For DeepSeek V4 -- \"minimal\", \"low\", \"medium\", \"high\", \"max\".

        Returns:
            LLMResponse with text, token counts, model name, and elapsed time.
        """
        ...
