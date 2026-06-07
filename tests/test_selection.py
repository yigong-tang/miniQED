import os, tempfile, pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.selection import run_selection


class MockSelectorAdapter:
    def __init__(self, selected="deepseek"):
        self.selected = selected

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text=f"SELECTED: {self.selected}\n\nReasoning: Best proof.",
            input_tokens=100,
            output_tokens=20,
            model="mock",
            elapsed_s=0.1,
        )


class TestRunSelection:
    @pytest.mark.asyncio
    async def test_selects_from_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            candidates = [
                {
                    "model": "deepseek",
                    "proof_file": os.path.join(d, "d.md"),
                    "verification_files": [],
                },
                {
                    "model": "opencode",
                    "proof_file": os.path.join(d, "o.md"),
                    "verification_files": [],
                },
            ]
            adapter = MockSelectorAdapter("deepseek")
            result = await run_selection(
                adapter=adapter,
                candidates=candidates,
                problem_file=os.path.join(d, "p.tex"),
                selection_file=os.path.join(d, "sel.md"),
                prompt_template_content="Select: {candidates_block}\nProblem: {problem_file}\nWrite to {selection_file}",
            )
            assert result == "deepseek"

    @pytest.mark.asyncio
    async def test_fallback_to_first(self):
        with tempfile.TemporaryDirectory() as d:
            candidates = [
                {
                    "model": "model-a",
                    "proof_file": os.path.join(d, "a.md"),
                    "verification_files": [],
                }
            ]
            adapter = MockSelectorAdapter("UNKNOWN")
            result = await run_selection(
                adapter=adapter,
                candidates=candidates,
                problem_file=os.path.join(d, "p.tex"),
                selection_file=os.path.join(d, "sel.md"),
                prompt_template_content="Select: {candidates_block}\nProblem: {problem_file}\nWrite to {selection_file}",
            )
            assert result == "model-a"
