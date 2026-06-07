"""Brainstorm step: multiple models independently generate proof strategies."""

import asyncio
import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.utils import load_prompt


async def run_brainstorm(
    *,
    adapters: list[AbstractLLMAdapter],
    providers: list[dict],
    problem_file: str,
    proof_file: str,
    related_info_dir: str,
    round_num: int,
    output_dir: str,
    prompt_template_path: str,
    prev_verification_dir: str = "",
) -> list[str]:
    """Run parallel brainstorm across multiple models."""
    os.makedirs(output_dir, exist_ok=True)
    prompts_dir = os.path.dirname(prompt_template_path)

    async def _brainstorm_single(index: int) -> str:
        provider = providers[index]
        adapter = adapters[index]
        name = provider.get("name", f"provider_{index}")
        output_file = os.path.join(output_dir, f"brainstorm_result_{name}.md")
        error_file = os.path.join(output_dir, f"error_brainstorm_{name}.md")

        prompt = load_prompt(
            prompts_dir, os.path.basename(prompt_template_path),
            problem_file=problem_file, related_info_dir=related_info_dir,
            proof_file=proof_file, prev_verification_dir=prev_verification_dir,
            round_num=round_num, output_file=output_file, error_file=error_file,
        )
        response = await adapter.chat(prompt=prompt)
        if not os.path.exists(output_file):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.text)
        return response.text

    tasks = [_brainstorm_single(i) for i in range(len(adapters))]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    texts = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[Brainstorm] Provider {i} failed: {r}")
        else:
            texts.append(r)
    return texts
