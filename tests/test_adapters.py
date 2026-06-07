import pytest
from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse


class TestAbstractLLMAdapter:
    def test_cannot_instantiate_abstract_adapter(self):
        """Abstract adapter should raise TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            AbstractLLMAdapter()  # noqa


class TestLLMResponse:
    def test_llm_response_creation(self):
        resp = LLMResponse(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            model="test-model",
            elapsed_s=1.5,
        )
        assert resp.text == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.model == "test-model"
        assert resp.elapsed_s == 1.5


import os
import asyncio
from mini_qed.adapters.openai_compatible import OpenAICompatibleAdapter


class TestOpenAICompatibleAdapter:
    @pytest.fixture
    def adapter(self):
        """Create adapter with environment variable or skip."""
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            pytest.skip("DEEPSEEK_API_KEY not set")
        return OpenAICompatibleAdapter(
            base_url="https://api.deepseek.com/v1",
            api_key=key,
            model="deepseek-v4-flash",
        )

    @pytest.mark.asyncio
    async def test_basic_chat(self, adapter):
        """A simple chat should return a non-empty response."""
        resp = await adapter.chat(
            prompt="Say exactly 'hello world' and nothing else.",
            temperature=0.0,
            max_tokens=50,
        )
        assert isinstance(resp.text, str)
        assert len(resp.text.strip()) > 0
        assert resp.model == "deepseek-v4-flash"
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0
        assert resp.elapsed_s > 0

    @pytest.mark.asyncio
    async def test_system_prompt(self, adapter):
        """System prompt should influence the response."""
        resp = await adapter.chat(
            prompt="What is your role?",
            system="You are a mathematician. Always answer 'I am a mathematician.'",
            temperature=0.0,
            max_tokens=100,
        )
        assert "mathematician" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_thinking_mode(self, adapter):
        """Thinking mode should not error."""
        resp = await adapter.chat(
            prompt="What is 2+2? Answer with just the number.",
            thinking=True,
            temperature=0.0,
            max_tokens=100,
        )
        assert "4" in resp.text
