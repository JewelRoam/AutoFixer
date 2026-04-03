"""Tests for autofixer.tools.git_miner — Git history distillation.

Tests that:
1. Bug-fix commits can be identified from git log
2. Commit diffs can be extracted as DiffPatch
3. Pre/post code context can be extracted as ContextSnapshot-like data
4. Results can be bulk-loaded into experience tensor format
"""

import os
import sys
import subprocess
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experience"))


@pytest.fixture
def git_repo_with_bugfix(tmp_path):
    """Create a minimal git repo with a 'fix' commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
           "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com"}

    def run(*args):
        subprocess.run(args, cwd=str(repo), env=env, check=True,
                       capture_output=True, text=True)

    run("git", "init")
    # Initial commit: buggy code
    (repo / "calc.py").write_text("def divide(a, b):\n    return a / b\n")
    run("git", "add", ".")
    run("git", "commit", "-m", "initial: add divide function")

    # Fix commit
    (repo / "calc.py").write_text(
        "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n"
    )
    run("git", "add", ".")
    run("git", "commit", "-m", "fix: handle division by zero in divide()")

    return str(repo)


class TestIdentifyBugfixCommits:
    def test_find_fix_commits(self, git_repo_with_bugfix):
        from autofixer.tools.git_miner import find_bugfix_commits
        commits = find_bugfix_commits(git_repo_with_bugfix)
        assert len(commits) >= 1
        assert any("fix" in c["message"].lower() for c in commits)

    def test_fix_commit_has_required_fields(self, git_repo_with_bugfix):
        from autofixer.tools.git_miner import find_bugfix_commits
        commits = find_bugfix_commits(git_repo_with_bugfix)
        for c in commits:
            assert "hash" in c
            assert "message" in c
            assert "author" in c


class TestExtractCommitDiff:
    def test_extract_diff_from_commit(self, git_repo_with_bugfix):
        from autofixer.tools.git_miner import find_bugfix_commits, extract_commit_diff
        commits = find_bugfix_commits(git_repo_with_bugfix)
        diff = extract_commit_diff(git_repo_with_bugfix, commits[0]["hash"])
        assert "---" in diff
        assert "+++" in diff
        assert "if b == 0:" in diff

    def test_extract_pre_commit_code(self, git_repo_with_bugfix):
        from autofixer.tools.git_miner import find_bugfix_commits, extract_pre_commit_code
        commits = find_bugfix_commits(git_repo_with_bugfix)
        pre_code = extract_pre_commit_code(
            git_repo_with_bugfix, commits[0]["hash"], "calc.py"
        )
        assert "return a / b" in pre_code
        assert "if b == 0" not in pre_code  # pre-fix version


class TestDistillToExperience:
    def test_distill_produces_experience_rows(self, git_repo_with_bugfix):
        from autofixer.tools.git_miner import distill_repo_to_experience_rows
        rows = distill_repo_to_experience_rows(git_repo_with_bugfix)
        assert len(rows) >= 1
        for row in rows:
            assert len(row) == 3  # [query, key, value]
            assert isinstance(row[0], str)  # query keywords
            assert isinstance(row[1], str)  # key (context)
            assert isinstance(row[2], str)  # value (diff patch)

    def test_distill_key_contains_code_context(self, git_repo_with_bugfix):
        from autofixer.tools.git_miner import distill_repo_to_experience_rows
        rows = distill_repo_to_experience_rows(git_repo_with_bugfix)
        # Key should contain pre-fix source code
        assert any("return a / b" in row[1] for row in rows)

    def test_distill_value_contains_diff(self, git_repo_with_bugfix):
        from autofixer.tools.git_miner import distill_repo_to_experience_rows
        rows = distill_repo_to_experience_rows(git_repo_with_bugfix)
        # Value should contain the fix diff
        assert any("if b == 0" in row[2] for row in rows)
