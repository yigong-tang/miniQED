import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.proof_search import ProofResult, run_proof_search


class MockAdapter:
    def __init__(self, response_text="Proof: QED."):
        self.response_text = response_text

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text=self.response_text,
            input_tokens=100,
            output_tokens=50,
            model="mock",
            elapsed_s=0.1,
        )


class TestRunProofSearch:
    @pytest.mark.asyncio
    async def test_produces_proof_file(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")
            with open(
                os.path.join(related_info_dir, "difficulty_evaluation.md"), "w"
            ) as f:
                f.write("Easy")
            with open(os.path.join(related_info_dir, "related_work.md"), "w") as f:
                f.write("# Work")

            # Create a minimal prompt template
            prompts_dir = os.path.join(d, "prompts")
            os.makedirs(prompts_dir)
            tmpl_path = os.path.join(prompts_dir, "proof_search.md")
            with open(tmpl_path, "w") as f:
                f.write(
                    "Problem: {problem_file}\n"
                    "Proof: {proof_file}\n"
                    "Related: {related_info_dir}\n"
                    "Round: {round_num}\n"
                    "Status: {proof_status_file}\n"
                    "Prev: {previous_round_instructions}\n"
                    "Human: {human_help_dir}\n"
                    "PrevHH: {prev_round_human_help_dir}\n"
                    "Skill: {skill_file}\n"
                    "Scratch: {scratch_pad_file}\n"
                    "Error: {error_file}\n"
                    "Brainstorm: {brainstorm_dir}"
                )

            adapter = MockAdapter("Proof: QED. Status: DONE.")
            result = await run_proof_search(
                adapter=adapter,
                model="mock",
                problem_file=problem_file,
                proof_file=os.path.join(d, "proof.md"),
                proof_status_file=os.path.join(d, "proof_status.md"),
                related_info_dir=related_info_dir,
                round_num=1,
                prev_instructions="- First round.",
                brainstorm_dir="",
                prompt_template_path=tmpl_path,
                prove_skill="",
            )
            assert result.proof_text == "Proof: QED. Status: DONE."
            assert os.path.exists(result.proof_file)

    @pytest.mark.asyncio
    async def test_proof_result_dataclass(self):
        """ProofResult stores all fields correctly."""
        r = ProofResult(
            proof_text="text",
            proof_file="proof.md",
            status_file="status.md",
            scratch_pad="scratch.md",
            model="mock",
            tokens_input=100,
            tokens_output=50,
        )
        assert r.proof_text == "text"
        assert r.proof_file == "proof.md"
        assert r.status_file == "status.md"
        assert r.scratch_pad == "scratch.md"
        assert r.model == "mock"
        assert r.tokens_input == 100
        assert r.tokens_output == 50

    @pytest.mark.asyncio
    async def test_saves_proof_file_when_missing(self):
        """Proof file is created when it does not exist."""
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")
            with open(
                os.path.join(related_info_dir, "difficulty_evaluation.md"), "w"
            ) as f:
                f.write("Medium")

            prompts_dir = os.path.join(d, "prompts")
            os.makedirs(prompts_dir)
            tmpl_path = os.path.join(prompts_dir, "proof_search.md")
            with open(tmpl_path, "w") as f:
                f.write("Problem: {problem_file}\nProof: {proof_file}")

            proof_file = os.path.join(d, "proof.md")
            assert not os.path.exists(proof_file)

            adapter = MockAdapter("A proof.")
            result = await run_proof_search(
                adapter=adapter,
                model="mock",
                problem_file=problem_file,
                proof_file=proof_file,
                proof_status_file=os.path.join(d, "proof_status.md"),
                related_info_dir=related_info_dir,
                round_num=1,
                prev_instructions="",
                brainstorm_dir="",
                prompt_template_path=tmpl_path,
                prove_skill="",
            )
            assert os.path.exists(proof_file)
            with open(proof_file) as f:
                content = f.read()
            assert content == "A proof."
            assert result.proof_text == "A proof."

    @pytest.mark.asyncio
    async def test_saves_status_file(self):
        """Status file is created with round info."""
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")

            prompts_dir = os.path.join(d, "prompts")
            os.makedirs(prompts_dir)
            tmpl_path = os.path.join(prompts_dir, "proof_search.md")
            with open(tmpl_path, "w") as f:
                f.write("Problem: {problem_file}\nProof: {proof_file}")

            status_file = os.path.join(d, "proof_status.md")
            adapter = MockAdapter("Status: proven.")
            result = await run_proof_search(
                adapter=adapter,
                model="mock",
                problem_file=problem_file,
                proof_file=os.path.join(d, "proof.md"),
                proof_status_file=status_file,
                related_info_dir=related_info_dir,
                round_num=2,
                prev_instructions="",
                brainstorm_dir="",
                prompt_template_path=tmpl_path,
                prove_skill="",
            )
            assert os.path.exists(status_file)
            with open(status_file, encoding="utf-8") as f:
                content = f.read()
            assert "# Proof Status" in content
            assert "Round 2" in content
            assert "Model: mock" in content

    @pytest.mark.asyncio
    async def test_copies_existing_proof_as_starting_point(self):
        """When proof_file exists, a proof_starting_point.md is created."""
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")

            prompts_dir = os.path.join(d, "prompts")
            os.makedirs(prompts_dir)
            tmpl_path = os.path.join(prompts_dir, "proof_search.md")
            with open(tmpl_path, "w") as f:
                f.write("Problem: {problem_file}\nProof: {proof_file}")

            proof_file = os.path.join(d, "proof.md")
            with open(proof_file, "w") as f:
                f.write("Old proof content")

            adapter = MockAdapter("New proof.")
            await run_proof_search(
                adapter=adapter,
                model="mock",
                problem_file=problem_file,
                proof_file=proof_file,
                proof_status_file=os.path.join(d, "proof_status.md"),
                related_info_dir=related_info_dir,
                round_num=1,
                prev_instructions="",
                brainstorm_dir="",
                prompt_template_path=tmpl_path,
                prove_skill="",
            )
            starting_point = os.path.join(d, "proof_starting_point.md")
            assert os.path.exists(starting_point)
            with open(starting_point) as f:
                content = f.read()
            assert content == "Old proof content"

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing_proof_file(self):
        """Existing proof file is NOT overwritten by the adapter response."""
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")

            prompts_dir = os.path.join(d, "prompts")
            os.makedirs(prompts_dir)
            tmpl_path = os.path.join(prompts_dir, "proof_search.md")
            with open(tmpl_path, "w") as f:
                f.write("Problem: {problem_file}\nProof: {proof_file}")

            proof_file = os.path.join(d, "proof.md")
            with open(proof_file, "w") as f:
                f.write("Existing proof content")

            adapter = MockAdapter("New proof content.")
            await run_proof_search(
                adapter=adapter,
                model="mock",
                problem_file=problem_file,
                proof_file=proof_file,
                proof_status_file=os.path.join(d, "proof_status.md"),
                related_info_dir=related_info_dir,
                round_num=1,
                prev_instructions="",
                brainstorm_dir="",
                prompt_template_path=tmpl_path,
                prove_skill="",
            )
            # Existing file with content should NOT be overwritten
            with open(proof_file) as f:
                content = f.read()
            assert content == "Existing proof content"
