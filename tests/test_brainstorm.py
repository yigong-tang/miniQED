import os, tempfile, pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.brainstorm import run_brainstorm


class MockBrainstormAdapter:
    def __init__(self, response_texts, model="mock"):
        self.response_texts = response_texts if isinstance(response_texts, list) else [response_texts]
        self.model = model
        self.call_count = 0

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        idx = self.call_count
        self.call_count += 1
        return LLMResponse(
            text=self.response_texts[idx % len(self.response_texts)],
            input_tokens=50, output_tokens=30, model=self.model, elapsed_s=0.1,
        )


class TestRunBrainstorm:
    @pytest.mark.asyncio
    async def test_parallel_brainstorm(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            proof_file = os.path.join(d, "proof.md")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write("test")

            prompts_dir = os.path.join(d, "prompts")
            os.makedirs(prompts_dir)
            tmpl = os.path.join(prompts_dir, "brainstorm.md")
            with open(tmpl, "w") as f:
                f.write(
                    "Problem: {problem_file}\nProof: {proof_file}\n"
                    "Related: {related_info_dir}\nRound: {round_num}\n"
                    "Output: {output_file}\nError: {error_file}\n"
                    "PrevVerif: {prev_verification_dir}"
                )

            adapters = [
                MockBrainstormAdapter("Use contradiction.", model="a"),
                MockBrainstormAdapter("Try induction.", model="b"),
            ]
            providers = [{"name": "a"}, {"name": "b"}]
            out = os.path.join(d, "brainstorm_results")
            os.makedirs(out)

            results = await run_brainstorm(
                adapters=adapters,
                providers=providers,
                problem_file=problem_file,
                proof_file=proof_file,
                related_info_dir=related_info_dir,
                round_num=1,
                output_dir=out,
                prompt_template_path=tmpl,
            )
            assert len(results) == 2
