"""Selection step: when multiple models generate proofs, pick the best one."""

import os
from mini_qed.adapters.base import AbstractLLMAdapter


async def run_selection(
    *, adapter: AbstractLLMAdapter, candidates: list[dict],
    problem_file: str, selection_file: str, prompt_template_content: str,
) -> str:
    """Select the best proof from multiple candidates."""
    lines = []
    for c in candidates:
        lines.append(f"**{c['model']}'s proof:** {c['proof_file']}")
        if c.get("verification_files"):
            lines.append(f"  Verification:")
            for vf in c["verification_files"]:
                lines.append(f"    - {vf}")
    candidates_block = "\n".join(lines)
    prompt = prompt_template_content.format(
        candidates_block=candidates_block, problem_file=problem_file,
        selection_file=selection_file)

    response = await adapter.chat(prompt=prompt)
    os.makedirs(os.path.dirname(selection_file), exist_ok=True)
    if not os.path.exists(selection_file):
        with open(selection_file, "w", encoding="utf-8") as f:
            f.write(response.text)

    for line in response.text.splitlines():
        if "SELECTED:" in line.upper():
            for c in candidates:
                if c["model"].lower() in line.lower():
                    return c["model"]
    return candidates[0]["model"] if candidates else "unknown"
