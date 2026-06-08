"""Top-level orchestrator: Stage 0 -> Stage 1 -> Stage 2."""

import argparse
import asyncio
import os
import shutil
import sys

from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.adapters.registry import AdapterRegistry
from mini_qed.config import load_pipeline_config
from mini_qed.logging import TokenTracker
from mini_qed.stages.stage1_simple import run_simple_proof_loop
from mini_qed.stages.stage2_summary import run_summary


async def run_pipeline(
    config_path: str, problem_file: str, output_dir: str | None = None
) -> bool:
    """Run the full miniQED pipeline. Returns True if proof verified."""
    pipeline_config, raw_config = load_pipeline_config(config_path)
    if output_dir:
        pipeline_config.output_dir = output_dir
    out = pipeline_config.output_dir

    registry = AdapterRegistry(raw_config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    prompts_dir = os.path.join(project_root, "prompts")
    skill_path = os.path.join(project_root, "skill", "super_math_skill.md")

    prove_skill = ""
    if os.path.exists(skill_path):
        with open(skill_path, encoding="utf-8") as f:
            prove_skill = f.read()

    # Copy human_help to output
    os.makedirs(os.path.join(out, "human_help"), exist_ok=True)
    hh_dir = os.path.join(project_root, "human_help")
    if os.path.isdir(hh_dir):
        for f_name in os.listdir(hh_dir):
            src = os.path.join(hh_dir, f_name)
            dst = os.path.join(out, "human_help", f_name)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    tracker = TokenTracker(out, pipeline_config.proof_summary.model)

    # Pre-load prompts
    prompts: dict[str, str] = {}
    for name in [
        "literature_survey",
        "proof_search",
        "proof_verify_structural",
        "proof_verify_detailed",
        "proof_verify_easy",
        "verdict_proof",
        "brainstorm",
        "proof_select",
        "proof_effort_summary",
    ]:
        path = os.path.join(prompts_dir, f"{name}.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                prompts[name] = f.read()

    # Stage 0: Skipped (frozen — literature survey requires tool calling, Phase 2)
    # Create minimal related_info for downstream compatibility
    related_info_dir = os.path.join(out, "related_info")
    os.makedirs(related_info_dir, exist_ok=True)
    difficulty_path = os.path.join(related_info_dir, "difficulty_evaluation.md")
    with open(difficulty_path, "w", encoding="utf-8") as f:
        f.write("## Classification: Medium\n\nAuto-classified. Literature survey frozen in Phase 1.\n")
    related_path = os.path.join(related_info_dir, "related_work.md")
    with open(related_path, "w", encoding="utf-8") as f:
        f.write("# Related Work\n\nLiterature survey is frozen in Phase 1. Proceeding with proof directly.\n")
    difficulty = "medium"
    print(f"\n  Difficulty: {difficulty} (literature survey frozen)\n")

    # Stage 1
    print(
        "\n"
        + "=" * 60
        + "\n  STAGE 1: Proof Search Loop (Simple Mode)\n"
        + "=" * 60
        + "\n"
    )
    sm = pipeline_config.simple_mode
    adapters: dict[str, AbstractLLMAdapter] = {}
    for role_cfg in [
        sm.proof_search,
        sm.structural_verifier,
        sm.detailed_verifier,
        sm.verdict,
    ]:
        if role_cfg.adapter not in adapters:
            adapters[role_cfg.adapter] = registry.get(role_cfg.adapter)
    # Multi-model adapters
    if sm.multi_model and sm.multi_model.get("enabled"):
        for prov in sm.multi_model.get("proof_search_providers", []):
            if prov["adapter"] not in adapters:
                adapters[prov["adapter"]] = registry.get(
                    prov["adapter"], model=prov.get("model")
                )
        for prov in sm.multi_model.get("verification_providers", []):
            if prov["adapter"] not in adapters:
                adapters[prov["adapter"]] = registry.get(
                    prov["adapter"], model=prov.get("model")
                )
    # Brainstorm adapters
    if sm.brainstorm and sm.brainstorm.get("enabled"):
        for prov in sm.brainstorm.get("providers", []):
            if prov["adapter"] not in adapters:
                adapters[prov["adapter"]] = registry.get(
                    prov["adapter"], model=prov.get("model")
                )

    success = await run_simple_proof_loop(
        config=pipeline_config,
        adapters=adapters,
        problem_file=problem_file,
        output_dir=out,
        prompts_dir=prompts_dir,
        related_info_dir=related_info_dir,
        prove_skill=prove_skill,
        tracker=tracker,
        difficulty=difficulty,
        prompts=prompts,
    )

    # Stage 2
    print(
        "\n" + "=" * 60 + "\n  STAGE 2: Proof Effort Summary\n" + "=" * 60 + "\n"
    )
    summary_cfg = pipeline_config.proof_summary
    summary_adapter = registry.get(summary_cfg.adapter, model=summary_cfg.model)
    await run_summary(
        adapter=summary_adapter,
        model=summary_cfg.model,
        output_dir=out,
        problem_file=problem_file,
        prompt_template_content=prompts["proof_effort_summary"],
        tracker=tracker,
    )

    print(f"\n{'=' * 60}")
    if success:
        print(f"  PROOF VERIFIED -- see {out}/proof.md")
    else:
        print(f"  MAX ITERATIONS REACHED -- see {out}/proof.md")
    print(
        f"  Token usage: {out}/TOKEN_USAGE.md\n{'=' * 60}\n"
    )
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="miniQED -- Mathematical Proof Pipeline"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default="problem/problem.tex")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--learn", action="store_true",
        help="After proof generation, launch interactive learning session"
    )
    parser.add_argument(
        "--learn-only", action="store_true",
        help="Skip proof generation, launch learning session with existing proof"
    )
    args = parser.parse_args()

    if args.learn_only:
        # Launch learning mode directly
        from mini_qed.learning.tutor import _main as learn_main
        sys.argv = [
            "tutor", "--config", args.config,
            "--problem", args.input,
        ]
        if args.output:
            sys.argv += ["--proof", f"{args.output}/proof.md",
                         "--summary", f"{args.output}/proof_effort_summary.md"]
        asyncio.run(learn_main())
        return

    success = asyncio.run(run_pipeline(args.config, args.input, args.output))

    if args.learn:
        # After pipeline, launch learning session
        pipeline_config, _ = load_pipeline_config(args.config)
        out = args.output or pipeline_config.output_dir
        from mini_qed.learning.tutor import _main as learn_main
        proof_path = os.path.join(out, "proof.md")
        summary_path = os.path.join(out, "proof_effort_summary.md")
        if os.path.exists(proof_path):
            sys.argv = [
                "tutor", "--config", args.config,
                "--problem", args.input,
                "--proof", proof_path,
                "--summary", summary_path,
            ]
            asyncio.run(learn_main())
        else:
            print("No proof.md found — skipping learning session.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
