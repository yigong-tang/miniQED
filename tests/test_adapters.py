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
