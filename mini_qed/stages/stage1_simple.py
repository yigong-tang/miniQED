"""Stage 1: Simple-mode proof loop -- the core orchestration module.

Ties together all step modules into a working proof loop:
    for round in 1..max_iterations:
        [optional: brainstorm] -> proof_search -> verification -> verdict -> DONE/CONTINUE
"""

import asyncio
import os
import shutil

from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.config import PipelineConfig
from mini_qed.logging import PipelineLogger, TokenTracker
from mini_qed.steps.brainstorm import run_brainstorm
from mini_qed.steps.proof_search import run_proof_search
from mini_qed.steps.selection import run_selection
from mini_qed.steps.verdict import Verdict, run_verdict
from mini_qed.steps.verification import (
    run_detailed_verification,
    run_easy_verification,
    run_structural_verification,
)
from mini_qed.utils import find_verification_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_difficulty(output_dir: str) -> str:
    """Read difficulty_evaluation.md and return the difficulty level.

    Scans for a line containing "classification" (case-insensitive).
    Returns ``"easy"``, ``"medium"", ``"hard"``, or ``"unknown"``.
    """
    diff_file = os.path.join(output_dir, "related_info", "difficulty_evaluation.md")
    if not os.path.isfile(diff_file):
        return "unknown"
    with open(diff_file, encoding="utf-8") as f:
        content = f.read()
    for line in content.splitlines():
        lower = line.strip().lower()
        if "classification" not in lower:
            continue
        if "easy" in lower:
            return "easy"
        if "medium" in lower:
            return "medium"
        if "hard" in lower:
            return "hard"
    return "unknown"


def _get_prompt_content(
    prompts: dict[str, str] | None,
    prompts_dir: str,
    name: str,
) -> str:
    """Return prompt content from the pre-loaded *prompts* dict, or load from file."""
    if prompts and name in prompts:
        return prompts[name]
    path = os.path.join(prompts_dir, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_prompt_file(
    prompts: dict[str, str] | None,
    prompts_dir: str,
    name: str,
    out_path: str,
) -> str:
    """Ensure prompt content is materialised as a file and return its path.

    When the content comes from the pre-loaded *prompts* dict it is written to
    *out_path* first.  Otherwise the existing file under *prompts_dir* is used.
    """
    if prompts and name in prompts:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prompts[name])
        return out_path
    return os.path.join(prompts_dir, name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_simple_proof_loop(
    *,
    config: PipelineConfig,
    adapters: dict[str, AbstractLLMAdapter],
    problem_file: str,
    output_dir: str,
    prompts_dir: str,
    related_info_dir: str,
    prove_skill: str,
    tracker: TokenTracker,
    start_round: int = 1,
    difficulty: str = "unknown",
    prompts: dict[str, str] | None = None,
) -> bool:
    """Run the simple-mode proof loop.

    Keyword-only parameters
    -----------------------
    config:
        Full pipeline configuration.
    adapters:
        Adapter instances keyed by name (matching config references).
    problem_file:
        Path to the problem statement (LaTeX).
    output_dir:
        Root output directory for all generated files.
    prompts_dir:
        Fallback directory for prompt template files (used when *prompts*
        does not contain a requested key).
    related_info_dir:
        Directory containing ``difficulty_evaluation.md`` etc.
    prove_skill:
        System prompt / skill text prepended to proof-search calls.
    tracker:
        Token-usage accumulator.
    start_round:
        Round number to begin at (default 1).
    difficulty:
        Pre-parsed difficulty; if ``"unknown"`` the function will try to
        determine it from ``difficulty_evaluation.md``.
    prompts:
        Pre-loaded prompt template content keyed by filename.
        Falls back to file loading from *prompts_dir* when a key is missing.

    Returns
    -------
    ``True`` if the proof loop finished with a **DONE** verdict,
    ``False`` if *max_proof_iterations* was reached without completion.
    """
    simple_cfg = config.simple_mode
    max_iter = config.max_proof_iterations

    # ---- Logger -----------------------------------------------------------
    log_dir = os.path.join(output_dir, "simple_proof_log")
    logger = PipelineLogger(log_dir, "Simple Proof Loop")
    logger.log(
        f"[Stage1] Starting simple proof loop "
        f"(max_iter={max_iter}, difficulty={difficulty})"
    )

    # ---- Difficulty -------------------------------------------------------
    if difficulty == "unknown":
        difficulty = _parse_difficulty(output_dir)
        logger.log(f"[Stage1] Parsed difficulty: {difficulty}")

    # ---- Load prompt templates (content or file fallback) -----------------
    prompt_proof_search = _get_prompt_content(prompts, prompts_dir, "proof_search.md")
    prompt_verify_structural = _get_prompt_content(
        prompts, prompts_dir, "proof_verify_structural.md",
    )
    prompt_verify_detailed = _get_prompt_content(
        prompts, prompts_dir, "proof_verify_detailed.md",
    )
    prompt_verify_easy = _get_prompt_content(
        prompts, prompts_dir, "proof_verify_easy.md",
    )
    prompt_verdict = _get_prompt_content(prompts, prompts_dir, "verdict_proof.md")
    prompt_brainstorm = _get_prompt_content(prompts, prompts_dir, "brainstorm.md")
    prompt_select = _get_prompt_content(prompts, prompts_dir, "proof_select.md")

    # ---- Config knobs -----------------------------------------------------
    brainstorm_cfg = simple_cfg.brainstorm or {}
    multimodel_cfg = simple_cfg.multi_model or {}
    brainstorm_enabled = brainstorm_cfg.get("enabled", False)
    multimodel_enabled = multimodel_cfg.get("enabled", False)

    # Path to the "live" proof file (carried forward between rounds)
    live_proof = os.path.join(output_dir, "proof.md")

    # ======================================================================
    # Round loop
    # ======================================================================
    for round_num in range(start_round, max_iter + 1):
        round_dir = os.path.join(output_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        logger.update_status(
            round_num, max_iter, "Proof Search", "RUNNING",
            f"Round {round_num}: preparing directories...",
        )
        logger.append_history(f"Round {round_num} started")
        logger.log(f"[Stage1] === Round {round_num}/{max_iter} ===")

        # ---- Backup proof from previous round -----------------------------
        round_proof = os.path.join(round_dir, "proof.md")
        if os.path.isfile(live_proof):
            shutil.copy2(live_proof, round_proof)

        # ---- Human-help directories ---------------------------------------
        human_help_global_dir = os.path.join(output_dir, "human_help")
        human_help_round_dir = os.path.join(round_dir, "human_help")
        os.makedirs(human_help_global_dir, exist_ok=True)
        os.makedirs(human_help_round_dir, exist_ok=True)

        if round_num > start_round:
            prev_round_human_help_dir = os.path.join(
                output_dir, f"round_{round_num - 1}", "human_help",
            )
        else:
            prev_round_human_help_dir = ""

        # ------------------------------------------------------------------
        # Optional brainstorm
        # ------------------------------------------------------------------
        brainstorm_dir = os.path.join(round_dir, "brainstorm")
        if brainstorm_enabled:
            logger.log(f"[Stage1] Round {round_num}: Brainstorm phase")
            brainstorm_providers = brainstorm_cfg.get("providers", [])
            b_adapters: list[AbstractLLMAdapter] = []
            for prov in brainstorm_providers:
                aname = prov.get("adapter", "")
                if aname in adapters:
                    b_adapters.append(adapters[aname])

            if b_adapters:
                b_prompt_path = _write_prompt_file(
                    prompts, prompts_dir, "brainstorm.md",
                    os.path.join(round_dir, "prompt_brainstorm.md"),
                )
                prev_verif_dir = (
                    os.path.join(output_dir, f"round_{round_num - 1}")
                    if round_num > start_round
                    else ""
                )
                await run_brainstorm(
                    adapters=b_adapters,
                    providers=brainstorm_providers,
                    problem_file=problem_file,
                    proof_file=round_proof,
                    related_info_dir=related_info_dir,
                    round_num=round_num,
                    output_dir=brainstorm_dir,
                    prompt_template_path=b_prompt_path,
                    prev_verification_dir=prev_verif_dir,
                )
            else:
                logger.log(
                    f"[Stage1] Round {round_num}: Brainstorm enabled but no "
                    f"adapters configured",
                )

        # Make brainstorm_dir available for proof-search below
        bs_dir_arg = brainstorm_dir if (
            brainstorm_enabled and os.path.isdir(brainstorm_dir)
        ) else ""

        # ------------------------------------------------------------------
        # Proof search
        # ------------------------------------------------------------------
        logger.update_status(
            round_num, max_iter, "Proof Search", "RUNNING",
            f"Round {round_num}: running proof search...",
        )
        logger.log(f"[Stage1] Round {round_num}: Proof search phase")

        if multimodel_enabled:
            multimodel_providers = multimodel_cfg.get("providers", [])
            tasks = []
            for prov in multimodel_providers:
                aname = prov.get("adapter", "")
                mname = prov.get("model", aname)
                if aname not in adapters:
                    continue

                model_proof_file = os.path.join(round_dir, f"proof_{mname}.md")
                model_status_file = os.path.join(
                    round_dir, f"proof_status_{mname}.md",
                )
                model_scratch = os.path.join(round_dir, f"scratch_pad_{mname}.md")
                model_error = os.path.join(
                    round_dir, f"error_proof_search_{mname}.md",
                )

                ps_prompt_path = _write_prompt_file(
                    prompts, prompts_dir, "proof_search.md",
                    os.path.join(round_dir, f"prompt_proof_search_{mname}.md"),
                )

                tasks.append(
                    run_proof_search(
                        adapter=adapters[aname],
                        model=mname,
                        problem_file=problem_file,
                        proof_file=model_proof_file,
                        proof_status_file=model_status_file,
                        related_info_dir=related_info_dir,
                        round_num=round_num,
                        prev_instructions="",
                        brainstorm_dir=bs_dir_arg,
                        prompt_template_path=ps_prompt_path,
                        prove_skill=prove_skill,
                        human_help_dir=human_help_global_dir,
                        prev_round_human_help_dir=prev_round_human_help_dir,
                        scratch_pad_file=model_scratch,
                        error_file=model_error,
                    )
                )

            if not tasks:
                logger.log(
                    f"[Stage1] Round {round_num}: Multi-model enabled but "
                    f"no valid providers -- falling back to single model",
                )
                multimodel_enabled = False  # fall through below
            else:
                proof_results = await asyncio.gather(*tasks)

                for pr in proof_results:
                    tracker.record(
                        f"Proof Search (multi:{pr.model})",
                        pr.tokens_input,
                        pr.tokens_output,
                        0,
                        model=pr.model,
                    )

                # ---- Selection -------------------------------------------
                logger.log(f"[Stage1] Round {round_num}: Selecting best proof")
                candidates = []
                for pr in proof_results:
                    candidates.append({
                        "model": pr.model,
                        "proof_file": pr.proof_file,
                        "verification_files": find_verification_files(
                            os.path.dirname(pr.proof_file),
                        ),
                    })

                selection_file = os.path.join(round_dir, "proof_selection.md")
                selected_model = await run_selection(
                    adapter=adapters[simple_cfg.proof_search.adapter],
                    candidates=candidates,
                    problem_file=problem_file,
                    selection_file=selection_file,
                    prompt_template_content=prompt_select,
                )
                logger.log(
                    f"[Stage1] Round {round_num}: Selected proof from "
                    f"{selected_model}",
                )

                selected_proof = None
                for pr in proof_results:
                    if pr.model == selected_model:
                        selected_proof = pr.proof_file
                        break
                if selected_proof:
                    shutil.copy2(selected_proof, round_proof)
                    shutil.copy2(selected_proof, live_proof)

        if not multimodel_enabled:
            # ---- Single-model proof search --------------------------------
            ps_adapter_name = simple_cfg.proof_search.adapter
            ps_model = simple_cfg.proof_search.model

            ps_prompt_path = _write_prompt_file(
                prompts, prompts_dir, "proof_search.md",
                os.path.join(round_dir, "prompt_proof_search.md"),
            )

            proof_result = await run_proof_search(
                adapter=adapters[ps_adapter_name],
                model=ps_model,
                problem_file=problem_file,
                proof_file=round_proof,
                proof_status_file=os.path.join(round_dir, "proof_status.md"),
                related_info_dir=related_info_dir,
                round_num=round_num,
                prev_instructions="",
                brainstorm_dir=bs_dir_arg,
                prompt_template_path=ps_prompt_path,
                prove_skill=prove_skill,
                human_help_dir=human_help_global_dir,
                prev_round_human_help_dir=prev_round_human_help_dir,
                scratch_pad_file=os.path.join(round_dir, "scratch_pad.md"),
                error_file=os.path.join(round_dir, "error_proof_search.md"),
            )

            tracker.record(
                "Proof Search",
                proof_result.tokens_input,
                proof_result.tokens_output,
                0,
                model=ps_model,
            )

            # Carry forward
            if os.path.isfile(round_proof):
                shutil.copy2(round_proof, live_proof)

        # ------------------------------------------------------------------
        # Verification
        # ------------------------------------------------------------------
        logger.update_status(
            round_num, max_iter, "Verification", "RUNNING",
            f"Round {round_num}: running verification...",
        )
        logger.log(
            f"[Stage1] Round {round_num}: Verification phase "
            f"(difficulty={difficulty})",
        )

        verification_dir = os.path.join(round_dir, "verification")
        os.makedirs(verification_dir, exist_ok=True)

        # Verifier adapters (may point to the same backend for both roles)
        struct_adapter_name = simple_cfg.structural_verifier.adapter
        detailed_adapter_name = simple_cfg.detailed_verifier.adapter

        structural_pass = True  # scoped for potential detailed re-use

        if difficulty == "easy":
            easy_reports = await run_easy_verification(
                adapters=[adapters[struct_adapter_name]],
                verifier_names=[struct_adapter_name],
                problem_file=problem_file,
                proof_file=round_proof,
                output_dir=verification_dir,
                prompt_template_content=prompt_verify_easy,
            )
            for r in easy_reports:
                tracker.record(
                    f"Verification (easy:{r.model})",
                    r.tokens_input, r.tokens_output, 0, model=r.model,
                )
        else:
            # ---- Structural verification (non-easy) -----------------------
            structural_dir = os.path.join(verification_dir, "structural")
            os.makedirs(structural_dir, exist_ok=True)

            structural_reports = await run_structural_verification(
                adapters=[adapters[struct_adapter_name]],
                verifier_names=[struct_adapter_name],
                problem_file=problem_file,
                proof_file=round_proof,
                output_dir=structural_dir,
                prompt_template_content=prompt_verify_structural,
            )
            for r in structural_reports:
                tracker.record(
                    f"Verification (structural:{r.model})",
                    r.tokens_input, r.tokens_output, 0, model=r.model,
                )

            structural_pass = all(
                r.verdict == "PASS" for r in structural_reports
            )
            logger.log(
                f"[Stage1] Round {round_num}: Structural "
                f"{'PASS' if structural_pass else 'FAIL'}",
            )

            if structural_pass:
                # ---- Detailed verification (only when structural passes) --
                detailed_dir = os.path.join(verification_dir, "detailed")
                os.makedirs(detailed_dir, exist_ok=True)

                detailed_reports = await run_detailed_verification(
                    adapters=[adapters[detailed_adapter_name]],
                    verifier_names=[detailed_adapter_name],
                    problem_file=problem_file,
                    proof_file=round_proof,
                    structural_report_file=os.path.join(
                        structural_dir, "verification_result.md",
                    ),
                    output_dir=detailed_dir,
                    prompt_template_content=prompt_verify_detailed,
                )
                for r in detailed_reports:
                    tracker.record(
                        f"Verification (detailed:{r.model})",
                        r.tokens_input, r.tokens_output, 0, model=r.model,
                    )

        # ------------------------------------------------------------------
        # Verdict
        # ------------------------------------------------------------------
        logger.update_status(
            round_num, max_iter, "Verdict", "RUNNING",
            f"Round {round_num}: running verdict...",
        )
        logger.log(f"[Stage1] Round {round_num}: Verdict phase")

        # Collect verification result files
        verification_files: list[str] = []
        if difficulty == "easy":
            verification_files = find_verification_files(verification_dir)
        else:
            structural_dir = os.path.join(verification_dir, "structural")
            detailed_dir = os.path.join(verification_dir, "detailed")
            verification_files = find_verification_files(structural_dir)
            if os.path.isdir(detailed_dir):
                verification_files.extend(
                    find_verification_files(detailed_dir),
                )

        if not verification_files:
            logger.log(
                f"[Stage1] Round {round_num}: No verification files found -- "
                f"verdict = CONTINUE",
            )
            verdict = Verdict.CONTINUE
        else:
            verdict = await run_verdict(
                adapter=adapters[simple_cfg.verdict.adapter],
                verification_files=verification_files,
                verdict_prompt_text=prompt_verdict,
            )

        logger.log(f"[Stage1] Round {round_num}: Verdict = {verdict.value}")
        logger.append_history(f"Round {round_num}: {verdict.value}")

        if verdict == Verdict.DONE:
            logger.update_status(
                round_num, max_iter, "DONE", "FINISHED",
                f"Proof completed at round {round_num}",
            )
            logger.log(f"[Stage1] Proof DONE at round {round_num}")
            logger.finalize(
                round_num, max_iter, "DONE",
                f"Proof completed at round {round_num}",
            )
            return True

    # ---- Max iterations exhausted -----------------------------------------
    logger.log(
        f"[Stage1] Max iterations ({max_iter}) reached without DONE verdict",
    )
    logger.update_status(
        max_iter, max_iter, "MAX_ITERATIONS", "FAILED",
        f"Reached max iterations ({max_iter}) without proof completion",
    )
    logger.finalize(
        max_iter, max_iter, "MAX_ITERATIONS",
        f"Reached max iterations ({max_iter})",
    )
    return False
