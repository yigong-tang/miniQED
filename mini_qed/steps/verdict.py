"""Verdict step: reads verification reports and decides DONE or CONTINUE."""

import enum
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.utils import load_prompt


class Verdict(enum.Enum):
    DONE = "DONE"
    CONTINUE = "CONTINUE"


async def run_verdict(
    *,
    adapter: AbstractLLMAdapter,
    verification_files: list[str],
    prompt_template_path: str = "",
    verdict_prompt_text: str = "",
) -> Verdict:
    """Run the verdict agent -- decide whether the proof is done."""
    if prompt_template_path:
        import os

        prompts_dir = os.path.dirname(prompt_template_path)
        prompt_name = os.path.basename(prompt_template_path)
        if len(verification_files) == 1:
            ref = f"Read the verification result file at `{verification_files[0]}`."
        else:
            ref = "\n".join(f"- `{f}`" for f in verification_files)
        prompt = load_prompt(prompts_dir, prompt_name, verification_result_file=ref)
    else:
        ref = "\n".join(verification_files)
        prompt = verdict_prompt_text.replace("{verification_files}", ref)

    response = await adapter.chat(prompt=prompt, temperature=0.0, max_tokens=100)

    text = response.text.strip().upper()
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped == "DONE":
            return Verdict.DONE
        if stripped == "CONTINUE":
            return Verdict.CONTINUE

    if "DONE" in text and "CONTINUE" not in text:
        return Verdict.DONE
    return Verdict.CONTINUE
