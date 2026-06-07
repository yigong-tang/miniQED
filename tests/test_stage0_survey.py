import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.stages.stage0_survey import run_literature_survey


class MockSurveyAdapter:
    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text="Difficulty: Easy. Use Rolle's theorem.",
            input_tokens=200, output_tokens=100, model="mock", elapsed_s=0.2,
        )


class TestStage0Survey:
    @pytest.mark.asyncio
    async def test_produces_expected_files(self):
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "problem.tex")
            with open(pf, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")
            rid = await run_literature_survey(
                adapter=MockSurveyAdapter(), model="mock", problem_file=pf,
                output_dir=d,
                prompt_template_content=(
                    "Survey: {problem_file}\n"
                    "Write difficulty to {difficulty_file}\n"
                    "Write related work to {related_work_file}\n"
                    "Error to {error_file}"
                ),
            )
            assert os.path.exists(os.path.join(rid, "difficulty_evaluation.md"))
            assert os.path.exists(os.path.join(rid, "related_work.md"))
