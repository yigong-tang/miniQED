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


from mini_qed.adapters.registry import build_adapter, AdapterRegistry
from mini_qed.adapters.openai_compatible import OpenAICompatibleAdapter
from mini_qed.adapters.anthropic_adapter import AnthropicAdapter
from mini_qed.adapters.openai_adapter import OpenAIAdapter


class TestAdapterRegistry:
    def test_build_openai_compatible_adapter(self):
        cfg = {
            "type": "openai_compatible",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-v4-flash",
        }
        adapter = build_adapter("deepseek", cfg)
        assert isinstance(adapter, OpenAICompatibleAdapter)

    def test_build_anthropic_adapter(self):
        cfg = {
            "type": "anthropic_sdk",
            "api_key": "sk-test",
            "model": "claude-sonnet-4-6",
        }
        adapter = build_adapter("claude", cfg)
        assert isinstance(adapter, AnthropicAdapter)

    def test_build_openai_adapter(self):
        cfg = {
            "type": "openai_sdk",
            "api_key": "sk-test",
            "model": "gpt-5.1",
        }
        adapter = build_adapter("openai", cfg)
        assert isinstance(adapter, OpenAIAdapter)

    def test_build_unknown_type_raises(self):
        cfg = {"type": "unknown_type", "api_key": "sk-test"}
        with pytest.raises(ValueError, match="Unknown adapter type"):
            build_adapter("bad", cfg)


class TestAdapterRegistryClass:
    def test_get_returns_correct_adapter(self):
        config = {
            "adapters": {
                "deepseek": {
                    "type": "openai_compatible",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "sk-test",
                },
                "claude": {
                    "type": "anthropic_sdk",
                    "api_key": "sk-test",
                },
            }
        }
        reg = AdapterRegistry(config)
        adapter = reg.get("deepseek", model="deepseek-v4-flash")
        assert isinstance(adapter, OpenAICompatibleAdapter)

    def test_get_missing_adapter_raises(self):
        reg = AdapterRegistry({"adapters": {}})
        with pytest.raises(ValueError, match="not found"):
            reg.get("nonexistent")
