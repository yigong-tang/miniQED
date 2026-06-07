"""Stage 2: Proof Effort Summary — reads all generated files and produces a summary."""

import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.logging import PipelineLogger


async def run_summary(
    *, adapter: AbstractLLMAdapter, model: str, output_dir: str,
    problem_file: str, prompt_template_content: str, tracker=None,
) -> str:
    """Run the summary agent. Returns path to proof_effort_summary.md."""
    log_dir = os.path.join(output_dir, "summary_log")
    logger = PipelineLogger(log_dir, "Summary")
    logger.update_status(1, 1, "Summary", "RUNNING", "Running summary agent...")

    summary_file = os.path.join(output_dir, "proof_effort_summary.md")
    error_file = os.path.join(log_dir, "error_summary.md")

    prompt = prompt_template_content.format(
        output_dir=output_dir, problem_file=problem_file,
        summary_file=summary_file, error_file=error_file)

    logger.log(f"[Summary] Starting (model={model})")
    response = await adapter.chat(prompt=prompt)
    logger.log(f"[Summary] Completed in {response.elapsed_s:.0f}s")

    if not os.path.exists(summary_file) and response.text.strip():
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.log("  Fallback: saved summary from response")

    if tracker:
        tracker.record("Proof Summary", response.input_tokens, response.output_tokens,
                       response.elapsed_s, provider="deepseek", model=model)

    logger.finalize(1, 1, "FINISHED", "Summary complete.")
    return summary_file
