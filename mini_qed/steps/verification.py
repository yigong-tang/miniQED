"""Verification steps: structural, detailed, and easy verification."""

import asyncio
import os
from dataclasses import dataclass
from mini_qed.adapters.base import AbstractLLMAdapter


@dataclass
class VerificationReport:
    verdict: str       # "PASS" or "FAIL"
    report_text: str
    report_file: str
    phase: str         # "structural", "detailed", "easy"
    model: str
    tokens_input: int
    tokens_output: int


def _verification_filename(verifier_name: str, multi_verifier: bool) -> str:
    if multi_verifier:
        return f"verification_result_{verifier_name}.md"
    return "verification_result.md"


async def _run_verification(
    *, adapters: list[AbstractLLMAdapter], verifier_names: list[str],
    prompt_factory, output_dir: str, phase: str, error_dir: str = "",
) -> list[VerificationReport]:
    multi = len(adapters) > 1
    error_dir = error_dir or output_dir

    async def _verify_one(index: int) -> VerificationReport | None:
        name = verifier_names[index]
        adapter = adapters[index]
        model = getattr(adapter, '_default_model', 'unknown')
        output_file = os.path.join(output_dir, _verification_filename(name, multi))
        error_file = os.path.join(error_dir, f"error_{phase}_{name}.md")
        prompt = prompt_factory(name, output_file, error_file)
        try:
            response = await adapter.chat(prompt=prompt)
        except Exception as e:
            os.makedirs(os.path.dirname(error_file), exist_ok=True)
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"# Verification Failed\n\n**Error:** {e}\n")
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if not os.path.exists(output_file):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.text)
        verdict = "FAIL"
        for line in response.text.splitlines():
            if "overall verdict" in line.lower():
                if "PASS" in line.upper():
                    verdict = "PASS"
                break
        return VerificationReport(verdict=verdict, report_text=response.text,
            report_file=output_file, phase=phase, model=model,
            tokens_input=response.input_tokens, tokens_output=response.output_tokens)

    results = await asyncio.gather(*[_verify_one(i) for i in range(len(adapters))])
    return [r for r in results if r is not None]


async def run_structural_verification(*, adapters, verifier_names, problem_file,
    proof_file, output_dir, prompt_template_content, additional_verify_rule_global_file="",
    additional_verify_rule_prev_round_file="") -> list[VerificationReport]:
    def _make_prompt(name, output_file, error_file):
        return prompt_template_content.format(
            problem_file=problem_file, proof_file=proof_file, output_file=output_file,
            error_file=error_file,
            additional_verify_rule_global_file=additional_verify_rule_global_file,
            additional_verify_rule_prev_round_file=additional_verify_rule_prev_round_file)
    return await _run_verification(adapters=adapters, verifier_names=verifier_names,
        prompt_factory=_make_prompt, output_dir=output_dir, phase="structural")


async def run_detailed_verification(*, adapters, verifier_names, problem_file,
    proof_file, structural_report_file, output_dir,
    prompt_template_content) -> list[VerificationReport]:
    def _make_prompt(name, output_file, error_file):
        return prompt_template_content.format(
            problem_file=problem_file, proof_file=proof_file,
            structural_report_file=structural_report_file,
            output_file=output_file, error_file=error_file)
    return await _run_verification(adapters=adapters, verifier_names=verifier_names,
        prompt_factory=_make_prompt, output_dir=output_dir, phase="detailed")


async def run_easy_verification(*, adapters, verifier_names, problem_file,
    proof_file, output_dir, prompt_template_content) -> list[VerificationReport]:
    def _make_prompt(name, output_file, error_file):
        return prompt_template_content.format(
            problem_file=problem_file, proof_file=proof_file,
            output_file=output_file, error_file=error_file)
    return await _run_verification(adapters=adapters, verifier_names=verifier_names,
        prompt_factory=_make_prompt, output_dir=output_dir, phase="easy")
