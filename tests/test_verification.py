import os, tempfile, pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.verification import run_structural_verification, run_easy_verification

class MockVerifierAdapter:
    def __init__(self, verdict="PASS", model="mock"):
        self.verdict = verdict; self.model = model
    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(text=f"## Overall Verdict: {self.verdict}\n\nThe proof is {'correct' if self.verdict == 'PASS' else 'incorrect'}.", input_tokens=100, output_tokens=50, model=self.model, elapsed_s=0.1)

class TestVerification:
    @pytest.mark.asyncio
    async def test_structural_verification_pass(self):
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "problem.tex"); prf = os.path.join(d, "proof.md")
            with open(pf, "w") as f: f.write("problem")
            with open(prf, "w") as f: f.write("proof")
            adapters = [MockVerifierAdapter("PASS")]
            results = await run_structural_verification(adapters=adapters, verifier_names=["deepseek"], problem_file=pf, proof_file=prf, output_dir=d, prompt_template_content="Verify: {problem_file} {proof_file}\nWrite to {output_file}")
            assert len(results) == 1; assert results[0].verdict == "PASS"

    @pytest.mark.asyncio
    async def test_easy_verification(self):
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "problem.tex"); prf = os.path.join(d, "proof.md")
            with open(pf, "w") as f: f.write("problem")
            with open(prf, "w") as f: f.write("proof")
            adapters = [MockVerifierAdapter("PASS")]
            results = await run_easy_verification(adapters=adapters, verifier_names=["deepseek"], problem_file=pf, proof_file=prf, output_dir=d, prompt_template_content="Easy: {problem_file} {proof_file}\nWrite to {output_file}")
            assert len(results) == 1; assert results[0].verdict == "PASS"
