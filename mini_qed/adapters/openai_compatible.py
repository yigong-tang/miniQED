"""OpenAI-compatible API adapter -- covers DeepSeek, GLM, MiniMax, Mimo, OpenCode."""

import time
from openai import AsyncOpenAI
from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse


class OpenAICompatibleAdapter(AbstractLLMAdapter):
    """Adapter for any LLM exposing an OpenAI-compatible /v1/chat/completions endpoint.

    Supports extended parameters for DeepSeek V4 (thinking, reasoning_effort).

    Usage:
        adapter = OpenAICompatibleAdapter(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model="deepseek-v4-pro",
        )
        resp = await adapter.chat("Prove that sqrt(2) is irrational.",
                                  thinking=True, reasoning_effort="max")
    """

    def __init__(self, base_url: str, api_key: str, model: str = ""):
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

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
        model = self._default_model

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # DeepSeek V4 extended parameters via extra_body
        extra: dict = {}
        if thinking:
            extra["thinking"] = {"type": "enabled"}
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        if extra:
            kwargs["extra_body"] = extra

        start = time.time()
        response = await self._client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=model,
            elapsed_s=elapsed,
        )
