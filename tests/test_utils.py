"""Tests for mini_qed.utils."""

import os
import pytest
from mini_qed.utils import load_prompt, file_nonempty, find_verification_files


# ---------------------------------------------------------------------------
# load_prompt
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_substitutes_placeholders(self, tmp_path):
        """Simple placeholders are replaced correctly."""
        prompt_file = tmp_path / "greeting.txt"
        prompt_file.write_text("Hello, {name}! You are {role}.")
        result = load_prompt(str(tmp_path), "greeting.txt", name="Alice", role="engineer")
        assert result == "Hello, Alice! You are engineer."

    def test_no_placeholders(self, tmp_path):
        """File with no placeholders is returned as-is."""
        prompt_file = tmp_path / "plain.txt"
        prompt_file.write_text("Just some text.")
        result = load_prompt(str(tmp_path), "plain.txt")
        assert result == "Just some text."

    def test_missing_placeholder_raises_key_error(self, tmp_path):
        """Missing kwarg for a {placeholder} raises KeyError."""
        prompt_file = tmp_path / "needs.txt"
        prompt_file.write_text("Hello {name}.")
        with pytest.raises(KeyError, match="name"):
            load_prompt(str(tmp_path), "needs.txt")

    def test_extra_kwargs_ignored(self, tmp_path):
        """Extra kwargs not in the template are ignored by format()."""
        prompt_file = tmp_path / "simple.txt"
        prompt_file.write_text("Only {a}.")
        result = load_prompt(str(tmp_path), "simple.txt", a="1", b="2")
        assert result == "Only 1."

    def test_file_not_found(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt(str(tmp_path), "nonexistent.txt")


# ---------------------------------------------------------------------------
# file_nonempty
# ---------------------------------------------------------------------------

class TestFileNonempty:
    def test_nonexistent_file(self):
        """Nonexistent file returns False."""
        assert file_nonempty("C:\\ definitely\\ does\\ not\\ exist\\ 12345") is False

    def test_empty_file(self, tmp_path):
        """Empty file returns False."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert file_nonempty(str(f)) is False

    def test_whitespace_only_file(self, tmp_path):
        """Whitespace-only file returns False (strip makes it empty)."""
        f = tmp_path / "whitespace.txt"
        f.write_text("   \n  \t  \n  ")
        assert file_nonempty(str(f)) is False

    def test_nonempty_file(self, tmp_path):
        """File with real content returns True."""
        f = tmp_path / "content.txt"
        f.write_text("hello world")
        assert file_nonempty(str(f)) is True


# ---------------------------------------------------------------------------
# find_verification_files
# ---------------------------------------------------------------------------

class TestFindVerificationFiles:
    def test_single_legacy_file(self, tmp_path):
        """Single verification_result.md is returned."""
        f = tmp_path / "verification_result.md"
        f.write_text("pass")
        result = find_verification_files(str(tmp_path))
        assert result == [str(f)]

    def test_multi_verifier_files(self, tmp_path):
        """Multiple verification_result_*.md files are returned sorted."""
        for name in ["verification_result_z.md", "verification_result_a.md"]:
            (tmp_path / name).write_text("pass")
        result = find_verification_files(str(tmp_path))
        expected = [
            str(tmp_path / "verification_result_a.md"),
            str(tmp_path / "verification_result_z.md"),
        ]
        assert result == expected

    def test_single_legacy_takes_precedence(self, tmp_path):
        """If legacy file exists, return only it even if multi files exist."""
        legacy = tmp_path / "verification_result.md"
        legacy.write_text("pass")
        multi = tmp_path / "verification_result_extra.md"
        multi.write_text("also pass")
        result = find_verification_files(str(tmp_path))
        assert result == [str(legacy)]

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        result = find_verification_files(str(tmp_path))
        assert result == []

    def test_nonexistent_directory(self):
        """Nonexistent directory returns empty list."""
        result = find_verification_files("C:\\ definitely\\ does\\ not\\ exist\\ 12345")
        assert result == []

    def test_empty_legacy_file_skipped(self, tmp_path):
        """Empty legacy file is not returned; falls through to multi files."""
        legacy = tmp_path / "verification_result.md"
        legacy.write_text("")
        multi = tmp_path / "verification_result_b.md"
        multi.write_text("real content")
        result = find_verification_files(str(tmp_path))
        assert result == [str(multi)]

    def test_empty_multi_files_skipped(self, tmp_path):
        """Empty multi-verifier files are not returned."""
        for name in ["verification_result_a.md", "verification_result_b.md"]:
            (tmp_path / name).write_text("")
        result = find_verification_files(str(tmp_path))
        assert result == []
