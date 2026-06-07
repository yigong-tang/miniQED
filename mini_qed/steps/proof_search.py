"""Single-model proof search step."""

import os
import shutil
from dataclasses import dataclass
from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse
from mini_qed.utils import load_prompt


@dataclass
class ProofResult:
    """Output of a single proof search call."""

    proof_text: str
    proof_file: str
    status_file: str
    scratch_pad: str
    model: str
    tokens_input: int
    tokens_output: int


async def run_proof_search(
    *,
    adapter: AbstractLLMAdapter,
    model: str,
    problem_file: str,
    proof_file: str,
    proof_status_file: str,
    related_info_dir: str,
    round_num: int,
    prev_instructions: str,
    brainstorm_dir: str,
    prompt_template_path: str,
    prove_skill: str,
    human_help_dir: str = "",
    prev_round_human_help_dir: str = "",
    scratch_pad_file: str = "",
    error_file: str = "",
) -> ProofResult:
    """Run a single proof search agent."""
    prompts_dir = os.path.dirname(prompt_template_path)
    prompt_name = os.path.basename(prompt_template_path)

    prompt = load_prompt(
        prompts_dir,
        prompt_name,
        problem_file=problem_file,
        proof_file=proof_file,
        related_info_dir=related_info_dir,
        round_num=round_num,
        proof_status_file=proof_status_file,
        previous_round_instructions=prev_instructions,
        human_help_dir=human_help_dir,
        prev_round_human_help_dir=prev_round_human_help_dir,
        skill_file=os.path.join(
            os.path.dirname(prompts_dir), "skill", "super_math_skill.md"
        ),
        scratch_pad_file=scratch_pad_file
        or os.path.join(os.path.dirname(proof_file), "scratch_pad.md"),
        error_file=error_file
        or os.path.join(os.path.dirname(proof_file), "error_proof_search.md"),
        brainstorm_dir=brainstorm_dir,
    )

    prompt += (
        f"\n\nThis is round {round_num}. Write or refine the proof. "
        "If one approach doesn't work after much effort, "
        "try a completely different proof strategy."
    )

    # Copy proof_file as starting point if exists
    if os.path.exists(proof_file):
        ctx_proof = os.path.join(
            os.path.dirname(proof_file), "proof_starting_point.md"
        )
        shutil.copy2(proof_file, ctx_proof)

    response: LLMResponse = await adapter.chat(prompt=prompt, system=prove_skill)

    # Save response to proof_file if agent didn't write it
    os.makedirs(os.path.dirname(proof_file), exist_ok=True)
    if not os.path.exists(proof_file) or os.path.getsize(proof_file) == 0:
        with open(proof_file, "w", encoding="utf-8") as f:
            f.write(response.text)

    # Save status file
    os.makedirs(os.path.dirname(proof_status_file), exist_ok=True)
    if not os.path.exists(proof_status_file) or os.path.getsize(proof_status_file) == 0:
        with open(proof_status_file, "w", encoding="utf-8") as f:
            f.write(f"# Proof Status — Round {round_num}\n\n")
            f.write(f"Model: {model}\n\n")
            f.write(f"## Response\n\n{response.text[:2000]}\n")

    return ProofResult(
        proof_text=response.text,
        proof_file=proof_file,
        status_file=proof_status_file,
        scratch_pad=scratch_pad_file or "",
        model=model,
        tokens_input=response.input_tokens,
        tokens_output=response.output_tokens,
    )
