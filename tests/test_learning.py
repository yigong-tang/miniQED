"""Tests for the interactive learning tutor."""

import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.learning.tutor import (
    _read_file,
    _build_system_prompt,
    _build_minimal_prompt,
    run_learning_session,
)


class TestReadFile:
    def test_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "test.txt")
            with open(p, "w") as f:
                f.write("hello world")
            assert _read_file(p) == "hello world"

    def test_returns_empty_for_missing_file(self):
        assert _read_file("/nonexistent/path.tex") == "(not available)"

    def test_returns_empty_for_empty_path(self):
        assert _read_file("") == "(not available)"


class TestBuildSystemPrompt:
    def test_includes_all_materials(self):
        tutor_prompt = "You are a tutor."
        problem = r"\begin{problem}x=1\end{problem}"
        proof = "Proof: trivial."
        summary = "It was easy."

        result = _build_system_prompt(tutor_prompt, problem, proof, summary)
        assert "You are a tutor." in result
        assert r"\begin{problem}x=1\end{problem}" in result
        assert "Proof: trivial." in result
        assert "It was easy." in result
        assert "Teaching Materials" in result
        assert "Never dump all of this" in result

    def test_truncates_long_proof(self):
        tutor_prompt = ""
        problem = ""
        proof = "x" * 10000
        summary = ""
        result = _build_system_prompt(tutor_prompt, problem, proof, summary)
        # Proof should be truncated to 8000 chars
        assert len(result) < 20000


class TestBuildMinimalPrompt:
    def test_contains_core_rules(self):
        prompt = _build_minimal_prompt()
        assert "Guide the user" in prompt
        assert "one step at a time" in prompt


class MockTutorAdapter:
    """Mock adapter that returns a pre-scripted tutoring response."""
    def __init__(self, responses=None):
        self.responses = responses or ["Let's begin. What do you know about this problem?"]
        self.call_count = 0

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        idx = self.call_count
        self.call_count += 1
        text = self.responses[idx % len(self.responses)]
        return LLMResponse(text=text, input_tokens=50, output_tokens=30, model="mock", elapsed_s=0.1)


class TestRunLearningSession:
    @pytest.mark.asyncio
    async def test_session_runs_without_error(self, monkeypatch):
        """Session should start, print opening, and exit on /quit."""
        adapter = MockTutorAdapter(["Welcome! What conditions do we have?"])

        # Simulate user typing /quit immediately
        inputs = iter(["/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        await run_learning_session(
            adapter=adapter,
            model="mock",
            problem="x = 1",
            proof="trivial",
            summary="easy",
            tutor_prompt="You are a tutor.",
        )
        # Should complete without exception
