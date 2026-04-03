"""Git history miner: distill bug-fix commits into experience rows.

Scans a git repository for commits with "fix", "bug", "patch", "resolve"
keywords, extracts pre/post code context and diffs, and produces
experience-compatible [query, key, value] rows.
"""

import os
import re
import subprocess
from typing import Dict, List, Optional


def _run_git(repo_path: str, *args: str) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", "-C", repo_path] + list(args),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


_FIX_KEYWORDS_RE = re.compile(
    r"\b(fix|bug|patch|resolve|hotfix|repair|workaround)\b", re.IGNORECASE
)


def find_bugfix_commits(repo_path: str) -> List[Dict[str, str]]:
    """Find commits whose message contains bug-fix keywords.

    Returns list of dicts with keys: hash, message, author.
    """
    log = _run_git(
        repo_path, "log", "--format=%H||%s||%an", "--all"
    )
    if not log:
        return []

    commits = []
    for line in log.splitlines():
        parts = line.split("||", 2)
        if len(parts) < 3:
            continue
        commit_hash, message, author = parts
        if _FIX_KEYWORDS_RE.search(message):
            commits.append({
                "hash": commit_hash,
                "message": message,
                "author": author,
            })
    return commits


def extract_commit_diff(repo_path: str, commit_hash: str) -> str:
    """Extract the unified diff for a given commit."""
    return _run_git(repo_path, "diff", f"{commit_hash}~1", commit_hash)


def extract_pre_commit_code(
    repo_path: str, commit_hash: str, filepath: str
) -> str:
    """Extract file content from the parent of the given commit."""
    return _run_git(repo_path, "show", f"{commit_hash}~1:{filepath}")


def _extract_changed_files(repo_path: str, commit_hash: str) -> List[str]:
    """Get list of files changed in a commit."""
    output = _run_git(
        repo_path, "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash
    )
    return [f for f in output.splitlines() if f.strip()]


def _generate_query_from_diff(diff_text: str, message: str) -> str:
    """Generate search keywords from a commit diff and message.

    Lightweight local extraction without LLM (for cold-start distillation).
    """
    keywords = set()

    # From commit message
    for word in re.findall(r"[a-zA-Z_]\w+", message):
        if len(word) > 2:
            keywords.add(word.lower())

    # From diff: added/removed lines
    for line in diff_text.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            for word in re.findall(r"[a-zA-Z_]\w+", line):
                if len(word) > 2:
                    keywords.add(word.lower())

    return "\n".join(sorted(keywords))


def distill_repo_to_experience_rows(repo_path: str) -> List[List[str]]:
    """Distill a git repo's bug-fix history into experience rows.

    Returns a list of [query, key, value] string triples where:
    - query: keywords extracted from commit message + diff
    - key: pre-fix source code context
    - value: the fix diff (unified format)
    """
    commits = find_bugfix_commits(repo_path)
    rows = []

    for commit in commits:
        diff = extract_commit_diff(repo_path, commit["hash"])
        if not diff:
            continue

        # Get pre-commit code for context (key)
        changed_files = _extract_changed_files(repo_path, commit["hash"])
        pre_code_parts = []
        for filepath in changed_files:
            try:
                pre = extract_pre_commit_code(repo_path, commit["hash"], filepath)
                if pre:
                    pre_code_parts.append(f"## {filepath}\n{pre}")
            except Exception:
                continue

        if not pre_code_parts:
            continue

        key = "\n\n".join(pre_code_parts)
        value = diff
        query = _generate_query_from_diff(diff, commit["message"])

        rows.append([query, key, value])

    return rows
