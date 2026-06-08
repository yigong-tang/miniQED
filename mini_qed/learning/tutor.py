"""Interactive guided learning tutor for mathematical proofs.

Uses a Socratic teaching method: the tutor has access to the problem statement,
the verified proof, and the proof journey summary, but reveals information
gradually through guiding questions rather than dumping the full proof.

Usage:
    python -m mini_qed.learning.tutor --problem problem/problem.tex \
        --proof proof_output/proof.md \
        --summary proof_output/proof_effort_summary.md
"""

import argparse
import asyncio
import os
import sys
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.adapters.registry import AdapterRegistry
from mini_qed.config import load_pipeline_config


def _read_file(path: str) -> str:
    """Read a file, return empty string if missing."""
    if not path or not os.path.exists(path):
        return "(not available)"
    with open(path, encoding="utf-8") as f:
        return f.read()


async def run_learning_session(
    *,
    adapter: AbstractLLMAdapter,
    model: str,
    problem: str,
    proof: str,
    summary: str = "",
    tutor_prompt: str = "",
) -> None:
    """Run an interactive Socratic tutoring session.

    Args:
        adapter: LLM adapter for the tutor agent.
        model: Model name (for display).
        problem: Content of problem.tex.
        proof: Content of proof.md.
        summary: Content of proof_effort_summary.md (optional).
        tutor_prompt: The tutor system prompt. If empty, uses a minimal built-in prompt.
    """
    if not tutor_prompt:
        tutor_prompt = _build_minimal_prompt()

    # Build the full system prompt with embedded materials
    system_prompt = _build_system_prompt(tutor_prompt, problem, proof, summary)

    # Phase 1: Let the tutor introduce the session
    print("\n" + "=" * 60)
    print("  miniQED  Interactive Learning Mode")
    print("=" * 60)
    print(f"  Tutor model: {model}")
    print(f"  Commands: /quit (exit), /hint (get a hint), /skip (reveal current step)")
    print("=" * 60 + "\n")

    opening_prompt = (
        "Begin Phase 1 (Problem Orientation). Introduce the problem in plain language, "
        "check the user's understanding of the conditions, and ask your first guiding question. "
        "Do NOT reveal the proof approach yet."
    )

    response = await adapter.chat(prompt=opening_prompt, system=system_prompt)
    print(f"🤖 Tutor: {response.text}\n")

    # Phase 2+: Interactive dialogue loop
    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("\n👋 Session ended. Great work today!")
            break

        if user_input.lower() == "/hint":
            user_input = "I'm stuck. Can you give me a hint without revealing the full step?"

        if user_input.lower() == "/skip":
            user_input = "I'd like to skip this step. Please reveal it and explain why it works, then ask me about the next step."

        # Build context reminder for the tutor
        context = (
            "Continue the guided learning session. Follow the teaching flow in your system prompt. "
            "If the user is stuck (/hint), give a small hint. "
            "If the user wants to skip (/skip), reveal the step with explanation, then move on. "
            "Remember: one question per turn. Guide, don't lecture."
        )

        response = await adapter.chat(prompt=user_input, system=system_prompt + "\n\n" + context)
        print(f"\n🤖 Tutor: {response.text}\n")

        # Track token usage
        if response.input_tokens > 0:
            print(f"   [{response.input_tokens} in / {response.output_tokens} out tokens]\n")


def _build_system_prompt(tutor_prompt: str, problem: str, proof: str, summary: str) -> str:
    """Build the complete system prompt with embedded materials."""
    materials = f"""
## Teaching Materials

### Problem Statement (LaTeX)
```
{problem.strip()}
```

### Complete Verified Proof
```
{proof.strip()[:8000]}
```

### Proof Journey Summary
```
{summary.strip()[:4000]}
```

---
Above are your reference materials. You know the full proof.
Your job is to guide the user to understand it, one step at a time.
Never dump all of this at once. Reveal strategically.
"""
    return tutor_prompt + "\n\n" + materials


def _build_minimal_prompt() -> str:
    """Fallback minimal tutor prompt if the main prompt file is unavailable."""
    return """You are a warm, encouraging peer math tutor.

Core rules:
1. Guide the user toward understanding through questions, not lectures.
2. You have access to the full proof. Reveal it one step at a time.
3. If the user is stuck after 2-3 attempts, give them the next step directly, then resume guiding.
4. End each response with exactly one question.
5. Be concise. 3-5 sentences per response."""


async def _main() -> None:
    """CLI entry point for standalone learning mode."""
    parser = argparse.ArgumentParser(description="miniQED Interactive Learning Tutor")
    parser.add_argument("--problem", default="problem/problem.tex", help="Path to problem.tex")
    parser.add_argument("--proof", default="proof_output/proof.md", help="Path to proof.md")
    parser.add_argument("--summary", default="proof_output/proof_effort_summary.md", help="Path to proof_effort_summary.md")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--provider", default="deepseek", help="Adapter name for the tutor")
    args = parser.parse_args()

    # Load config
    pipeline_config, raw_config = load_pipeline_config(args.config)
    registry = AdapterRegistry(raw_config)

    # Resolve tutor adapter — use verdict's adapter config as default since it uses flash (cheaper for chat)
    tutor_cfg = pipeline_config.simple_mode.verdict
    adapter = registry.get(tutor_cfg.adapter, model=tutor_cfg.model)

    # Load materials
    problem = _read_file(args.problem)
    proof = _read_file(args.proof)
    summary = _read_file(args.summary)

    if not problem.strip():
        print(f"ERROR: Problem file not found or empty: {args.problem}")
        sys.exit(1)
    if not proof.strip():
        print(f"WARNING: Proof file not found or empty: {args.proof}")
        print("Learning mode works best with an existing proof. Run the pipeline first.")

    # Load tutor prompt template
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prompt_path = os.path.join(project_root, "mini_qed", "learning", "tutor_prompt.md")
    tutor_prompt = _read_file(prompt_path) if os.path.exists(prompt_path) else ""

    await run_learning_session(
        adapter=adapter,
        model=tutor_cfg.model,
        problem=problem,
        proof=proof,
        summary=summary,
        tutor_prompt=tutor_prompt,
    )


if __name__ == "__main__":
    asyncio.run(_main())
