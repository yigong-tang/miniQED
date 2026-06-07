import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.stages.stage2_summary import run_summary


class MockSummaryAdapter:
    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text="# Proof Effort Summary\n\nDone.",
            input_tokens=100, output_tokens=50, model="mock", elapsed_s=0.1,
        )


class TestStage2Summary:
    @pytest.mark.asyncio
    async def test_produces_summary_file(self):
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "problem.tex")
            with open(pf, "w") as f:
                f.write("problem")
            sf = await run_summary(
                adapter=MockSummaryAdapter(), model="mock", output_dir=d,
                problem_file=pf,
                prompt_template_content=(
                    "Summarize:\n"
                    "Output: {output_dir}\n"
                    "Problem: {problem_file}\n"
                    "Write to {summary_file}\n"
                    "Error: {error_file}"
                ),
            )
            assert os.path.exists(sf)
