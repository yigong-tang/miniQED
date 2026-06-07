"""Shared utility functions."""

import os


def load_prompt(prompts_dir: str, filename: str, **kwargs) -> str:
    """Load a prompt template and fill {placeholders}.
    Raises FileNotFoundError if file missing, KeyError if placeholder has no kwarg."""
    path = os.path.join(prompts_dir, filename)
    with open(path, encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)


def file_nonempty(path: str) -> bool:
    """Return True if path exists and has non-whitespace content."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return bool(f.read().strip())


def find_verification_files(directory: str) -> list[str]:
    """Find all verification result files in a directory."""
    single = os.path.join(directory, "verification_result.md")
    if file_nonempty(single):
        return [single]
    files = []
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.startswith("verification_result_") and name.endswith(".md"):
                path = os.path.join(directory, name)
                if file_nonempty(path):
                    files.append(path)
    return files
