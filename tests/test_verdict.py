import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.verdict import run_verdict, Verdict


class MockVerdictAdapter:
    def __init__(self, decision: str):
        self.decision = decision

    async def chat(self, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(
            text=f"Overall verdict: {self.decision}",
            input_tokens=100, output_tokens=10, model="mock", elapsed_s=0.1,
        )


class TestRunVerdict:
    @pytest.mark.asyncio
    async def test_done_verdict(self):
        adapter = MockVerdictAdapter("DONE")
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "verification_result.md")
            with open(vf, "w") as f: f.write("Overall Verdict: PASS")
            result = await run_verdict(
                adapter=adapter, verification_files=[vf],
                verdict_prompt_text="Files: {verification_files}. Reply DONE or CONTINUE.",
            )
            assert result == Verdict.DONE

    @pytest.mark.asyncio
    async def test_continue_verdict(self):
        adapter = MockVerdictAdapter("CONTINUE")
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "verification_result.md")
            with open(vf, "w") as f: f.write("Overall Verdict: FAIL")
            result = await run_verdict(
                adapter=adapter, verification_files=[vf],
                verdict_prompt_text="Files: {verification_files}. Reply DONE or CONTINUE.",
            )
            assert result == Verdict.CONTINUE

    @pytest.mark.asyncio
    async def test_done_in_line(self):
        adapter = MockVerdictAdapter("the proof is DONE")
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "verification_result.md")
            with open(vf, "w") as f: f.write("OK")
            result = await run_verdict(
                adapter=adapter, verification_files=[vf],
                verdict_prompt_text="Files: {verification_files}. Reply.",
            )
            assert result == Verdict.DONE
