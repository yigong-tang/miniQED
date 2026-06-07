"""Tests for mini_qed/stages/stage1_simple.py."""

import os
import tempfile

import pytest

from mini_qed.adapters.base import LLMResponse
from mini_qed.stages.stage1_simple import _parse_difficulty, run_simple_proof_loop


# ===================================================================
# _parse_difficulty
# ===================================================================

class TestParseDifficulty:
    def test_easy(self):
        with tempfile.TemporaryDirectory() as d:
            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Easy")
            assert _parse_difficulty(d) == "easy"

    def test_medium(self):
        with tempfile.TemporaryDirectory() as d:
            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Medium")
            assert _parse_difficulty(d) == "medium"

    def test_hard(self):
        with tempfile.TemporaryDirectory() as d:
            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Hard")
            assert _parse_difficulty(d) == "hard"

    def test_unknown_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            assert _parse_difficulty(d) == "unknown"

    def test_unknown_when_no_classification_line(self):
        with tempfile.TemporaryDirectory() as d:
            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Random content\nNo classification here.")
            assert _parse_difficulty(d) == "unknown"

    def test_case_insensitive(self):
        with tempfile.TemporaryDirectory() as d:
            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## CLASSIFICATION: EASY")
            assert _parse_difficulty(d) == "easy"

    def test_easy_embedded_in_word(self):
        """'easy' is a substring check so 'uneasy' would also match if on the
        classification line -- accept this behaviour (the line will normally
        contain just the level)."""
        with tempfile.TemporaryDirectory() as d:
            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: VeryEasy")
            assert _parse_difficulty(d) == "easy"


# ===================================================================
# run_simple_proof_loop (integration-lite smoke test)
# ===================================================================

class MockLLMAdapter:
    """A minimal adapter that returns canned responses."""

    def __init__(self, name: str = "mock"):
        self._default_model = name

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text="# Proof\n\nThis is a proof.",
            input_tokens=100,
            output_tokens=50,
            model=self._default_model,
            elapsed_s=0.1,
        )


class MockTracker:
    def __init__(self):
        self.calls = []

    def record(self, call_name, input_tokens, output_tokens, elapsed,
               provider="deepseek", model=""):
        self.calls.append({
            "name": call_name,
            "input": input_tokens,
            "output": output_tokens,
        })


class TestRunSimpleProofLoop:
    """Smoke tests for the main orchestration loop.

    Because the loop calls many async sub-steps, we test that the basic
    scaffolding works: directories are created, the loop runs, and the
    result is a boolean.
    """

    @pytest.mark.asyncio
    async def test_non_easy_loop_runs_and_returns_false(self):
        """Medium-difficulty proof: structural + detailed verification path."""
        from mini_qed.config import (
            AgentRoleConfig,
            PipelineConfig,
            SimpleModeConfig,
        )

        class _VerifMock(MockLLMAdapter):
            """Mock that returns a PASS verdict for verification."""
            async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
                return LLMResponse(
                    text="# Verification Report\n\n"
                         "**Overall Verdict:** PASS\n\nProof looks correct.",
                    input_tokens=100,
                    output_tokens=50,
                    model=self._default_model,
                    elapsed_s=0.1,
                )

        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "problem.tex")
            with open(pf, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")

            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Medium")

            # Minimal prompts (one per required prompt file)
            pd = os.path.join(d, "prompts")
            os.makedirs(pd)
            for fname, content in [
                ("proof_search.md", "prove {problem_file} round {round_num}"),
                ("proof_verify_structural.md",
                 "struct check {problem_file} {proof_file}"),
                ("proof_verify_detailed.md",
                 "detail check {problem_file} {proof_file}"),
                ("proof_verify_easy.md",
                 "easy check {problem_file} {proof_file}"),
                ("verdict_proof.md",
                 "Verdict: {verification_files}"),
                ("brainstorm.md",
                 "Brainstorm {problem_file}"),
                ("proof_select.md",
                 "Select from {candidates_block}"),
            ]:
                with open(os.path.join(pd, fname), "w") as f:
                    f.write(content)

            skill = os.path.join(d, "skill")
            os.makedirs(skill)
            with open(os.path.join(skill, "super_math_skill.md"), "w") as f:
                f.write("You are a math expert.")

            config = PipelineConfig(
                max_proof_iterations=2,
                output_dir=d,
                literature_survey=AgentRoleConfig(adapter="mock", model="mock"),
                simple_mode=SimpleModeConfig(
                    proof_search=AgentRoleConfig(adapter="mock", model="mock"),
                    structural_verifier=AgentRoleConfig(
                        adapter="mock", model="mock",
                    ),
                    detailed_verifier=AgentRoleConfig(
                        adapter="mock", model="mock",
                    ),
                    verdict=AgentRoleConfig(adapter="mock", model="mock"),
                ),
                proof_summary=AgentRoleConfig(adapter="mock", model="mock"),
            )

            tracker = MockTracker()

            result = await run_simple_proof_loop(
                config=config,
                adapters={"mock": _VerifMock()},
                problem_file=pf,
                output_dir=d,
                prompts_dir=pd,
                related_info_dir=ri,
                prove_skill=os.path.join(skill, "super_math_skill.md"),
                tracker=tracker,
            )

            # Max iterations (2) reached without DONE verdict
            assert result is False
            # Round directories exist
            assert os.path.isdir(os.path.join(d, "round_1"))
            assert os.path.isdir(os.path.join(d, "round_2"))
            # Structural and detailed verification dirs exist in round_1
            v1 = os.path.join(d, "round_1", "verification", "structural")
            v2 = os.path.join(d, "round_1", "verification", "detailed")
            assert os.path.isdir(v1), f"Missing structural dir: {v1}"
            assert os.path.isdir(v2), f"Missing detailed dir: {v2}"

    @pytest.mark.asyncio
    async def test_easy_loop_one_phase_verification(self):
        """Easy difficulty: only easy verification runs (no structural/detailed)."""
        from mini_qed.config import (
            AgentRoleConfig,
            PipelineConfig,
            SimpleModeConfig,
        )

        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "problem.tex")
            with open(pf, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")

            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Easy")

            pd = os.path.join(d, "prompts")
            os.makedirs(pd)
            for fname, content in [
                ("proof_search.md", "prove {problem_file} round {round_num}"),
                ("proof_verify_structural.md",
                 "struct check {problem_file} {proof_file}"),
                ("proof_verify_detailed.md",
                 "detail check {problem_file} {proof_file}"),
                ("proof_verify_easy.md",
                 "easy check {problem_file} {proof_file}"),
                ("verdict_proof.md",
                 "Verdict: {verification_files}"),
                ("brainstorm.md",
                 "Brainstorm {problem_file}"),
                ("proof_select.md",
                 "Select from {candidates_block}"),
            ]:
                with open(os.path.join(pd, fname), "w") as f:
                    f.write(content)

            skill = os.path.join(d, "skill")
            os.makedirs(skill)
            with open(os.path.join(skill, "super_math_skill.md"), "w") as f:
                f.write("You are a math expert.")

            config = PipelineConfig(
                max_proof_iterations=1,
                output_dir=d,
                literature_survey=AgentRoleConfig(adapter="mock", model="mock"),
                simple_mode=SimpleModeConfig(
                    proof_search=AgentRoleConfig(adapter="mock", model="mock"),
                    structural_verifier=AgentRoleConfig(
                        adapter="mock", model="mock",
                    ),
                    detailed_verifier=AgentRoleConfig(
                        adapter="mock", model="mock",
                    ),
                    verdict=AgentRoleConfig(adapter="mock", model="mock"),
                ),
                proof_summary=AgentRoleConfig(adapter="mock", model="mock"),
            )

            mock = MockLLMAdapter()
            tracker = MockTracker()

            result = await run_simple_proof_loop(
                config=config,
                adapters={"mock": mock},
                problem_file=pf,
                output_dir=d,
                prompts_dir=pd,
                related_info_dir=ri,
                prove_skill=os.path.join(skill, "super_math_skill.md"),
                tracker=tracker,
            )

            assert result is False  # No DONE verdict possible with mock

            # Easy verification should NOT have structural/detailed subdirs
            v_dir = os.path.join(d, "round_1", "verification")
            assert os.path.isdir(v_dir)
            assert not os.path.isdir(os.path.join(v_dir, "structural"))
            assert not os.path.isdir(os.path.join(v_dir, "detailed"))

            # But easy verification output file should exist
            verif_files = []
            if os.path.isdir(v_dir):
                for fn in os.listdir(v_dir):
                    if fn.startswith("verification_result") and fn.endswith(".md"):
                        verif_files.append(fn)
            assert len(verif_files) >= 1, (
                f"Expected at least one verification_result file in {v_dir}, "
                f"got {verif_files}"
            )

    @pytest.mark.asyncio
    async def test_start_round_2_uses_existing_proof(self):
        """Starting at round 2 should backup the existing proof.md."""
        from mini_qed.config import (
            AgentRoleConfig,
            PipelineConfig,
            SimpleModeConfig,
        )

        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "problem.tex")
            with open(pf, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")

            ri = os.path.join(d, "related_info")
            os.makedirs(ri)
            with open(os.path.join(ri, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Medium")

            pd = os.path.join(d, "prompts")
            os.makedirs(pd)
            for fname, content in [
                ("proof_search.md", "prove {problem_file} round {round_num}"),
                ("proof_verify_structural.md",
                 "struct check {problem_file} {proof_file}"),
                ("proof_verify_detailed.md",
                 "detail check {problem_file} {proof_file}"),
                ("proof_verify_easy.md",
                 "easy check {problem_file} {proof_file}"),
                ("verdict_proof.md",
                 "Verdict: {verification_files}"),
                ("brainstorm.md",
                 "Brainstorm {problem_file}"),
                ("proof_select.md",
                 "Select from {candidates_block}"),
            ]:
                with open(os.path.join(pd, fname), "w") as f:
                    f.write(content)

            skill = os.path.join(d, "skill")
            os.makedirs(skill)
            with open(os.path.join(skill, "super_math_skill.md"), "w") as f:
                f.write("You are a math expert.")

            # Pre-seed proof.md
            with open(os.path.join(d, "proof.md"), "w") as f:
                f.write("# Existing proof draft")

            config = PipelineConfig(
                max_proof_iterations=3,
                output_dir=d,
                literature_survey=AgentRoleConfig(adapter="mock", model="mock"),
                simple_mode=SimpleModeConfig(
                    proof_search=AgentRoleConfig(adapter="mock", model="mock"),
                    structural_verifier=AgentRoleConfig(
                        adapter="mock", model="mock",
                    ),
                    detailed_verifier=AgentRoleConfig(
                        adapter="mock", model="mock",
                    ),
                    verdict=AgentRoleConfig(adapter="mock", model="mock"),
                ),
                proof_summary=AgentRoleConfig(adapter="mock", model="mock"),
            )

            mock = MockLLMAdapter()
            tracker = MockTracker()

            result = await run_simple_proof_loop(
                config=config,
                adapters={"mock": mock},
                problem_file=pf,
                output_dir=d,
                prompts_dir=pd,
                related_info_dir=ri,
                prove_skill=os.path.join(skill, "super_math_skill.md"),
                tracker=tracker,
                start_round=2,
            )

            assert result is False
            # Round 2 was created (because start_round=2)
            assert os.path.isdir(os.path.join(d, "round_2"))
            # Round 2 proof.md should exist (from backup of output_dir/proof.md)
            assert os.path.isfile(os.path.join(d, "round_2", "proof.md"))
