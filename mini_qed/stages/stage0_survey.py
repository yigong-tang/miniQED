"""Stage 0: Literature Survey -- evaluates problem difficulty and researches related work."""

import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.logging import PipelineLogger


async def run_literature_survey(
    *, adapter: AbstractLLMAdapter, model: str, problem_file: str,
    output_dir: str, prompt_template_content: str, prove_skill: str = "",
    tracker=None,
) -> str:
    """Run the literature survey agent. Returns path to related_info directory."""
    related_info_dir = os.path.join(output_dir, "related_info")
    os.makedirs(related_info_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "literature_survey_log")
    logger = PipelineLogger(log_dir, "Literature Survey")
    logger.update_status(1, 1, "Literature Survey", "RUNNING",
                         "Running literature survey agent...")

    difficulty_file = os.path.join(related_info_dir, "difficulty_evaluation.md")
    related_work_file = os.path.join(related_info_dir, "related_work.md")
    error_file = os.path.join(related_info_dir, "error_literature_survey.md")

    prompt = prompt_template_content.format(
        problem_file=problem_file, difficulty_file=difficulty_file,
        related_work_file=related_work_file, error_file=error_file)

    logger.log(f"[Survey] Starting (model={model})")
    response = await adapter.chat(prompt=prompt, system=prove_skill)
    logger.log(f"[Survey] Completed in {response.elapsed_s:.0f}s")

    # Fallback: save if agent didn't write files
    for path, label in [(difficulty_file, "difficulty"), (related_work_file, "related work")]:
        if not os.path.exists(path) and response.text.strip():
            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)
            logger.log(f"  Fallback: saved {label} from response")

    for path, label in [(difficulty_file, "difficulty evaluation"), (related_work_file, "related work")]:
        if not os.path.exists(path):
            msg = f"FATAL -- Literature Survey: expected file missing: {label} ({path})"
            logger.log(msg)
            raise RuntimeError(msg)

    if not os.path.exists(error_file):
        with open(error_file, "w", encoding="utf-8") as f:
            f.write("")

    if tracker:
        tracker.record("Literature Survey", response.input_tokens, response.output_tokens,
                       response.elapsed_s, provider="deepseek", model=model)

    logger.finalize(1, 1, "FINISHED", "Literature survey complete.")
    return related_info_dir
