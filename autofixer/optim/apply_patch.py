"""Apply patch to workspace and append experience to tensor.

Handles:
1. Applying a unified diff patch to a real file on disk
2. Appending a new (query, key, value) experience row to the experience tensor
"""

import os
import re
import subprocess
import tempfile
from typing import Optional

import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

from autofixer.context_snapshot import ContextSnapshot


def _realpath(path: str) -> str:
    """Normalize path to avoid macOS /var vs /private/var symlink issues."""
    return os.path.realpath(path)


def apply_patch_to_file(filepath: str, patch_text: str) -> bool:
    """Apply a unified diff patch to a file on disk.

    Args:
        filepath: Absolute path to the file to patch.
        patch_text: Unified diff text (diff -u format).

    Returns:
        True if patch applied successfully, False otherwise.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False
    ) as pf:
        pf.write(patch_text)
        patch_file = pf.name

    try:
        result = subprocess.run(
            ["patch", "--batch", "--fuzz=2", filepath, patch_file],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(patch_file)


def _generate_query_keywords(snapshot: ContextSnapshot, diff_patch: str) -> str:
    """Generate query keywords from a snapshot and patch for experience retrieval.

    Extracts salient keywords from exception type, variable names, and patch content.
    This is a lightweight local extraction — no LLM call needed for cold-start.
    """
    keywords = set()

    # Exception type is always a key signal
    keywords.add(snapshot.exception_type)

    # Extract identifiers from source code (simple heuristic)
    for word in re.findall(r"[a-zA-Z_]\w+", snapshot.source_code):
        if len(word) > 2:
            keywords.add(word)

    # Extract from diff patch
    for line in diff_patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            for word in re.findall(r"[a-zA-Z_]\w+", line):
                if len(word) > 2:
                    keywords.add(word)

    return "\n".join(sorted(keywords))


def append_experience(
    experience_tensor: torch.Tensor,
    snapshot: ContextSnapshot,
    diff_patch: str,
    relative_to: str,
) -> torch.Tensor:
    """Append a new experience row to the experience tensor.

    Creates a [N+1, 3] tensor, copies old experience via slice_view + assign_tensor,
    then writes the new row into the last slot.

    Args:
        experience_tensor: Existing [N, 3] experience tensor.
        snapshot: The crash context that was fixed.
        diff_patch: The successful fix patch (diff -u format).
        relative_to: Directory for tensor storage.

    Returns:
        New [N+1, 3] experience tensor.
    """
    # Normalize path to canonical form (avoids macOS /var vs /private/var issues)
    relative_to = _realpath(relative_to)

    query = _generate_query_keywords(snapshot, diff_patch)
    key = snapshot.to_text()
    value = diff_patch

    new_row = make_tensor([[query, key, value]], relative_to)

    old_n = experience_tensor.shape[0]
    new_n = old_n + 1

    # Allocate new [N+1, 3] tensor with real files (empty strings) so symlinks resolve
    placeholder = [[""] * 3 for _ in range(new_n)]
    result = make_tensor(placeholder, relative_to)

    # Copy existing rows via slice_view + assign_tensor
    if old_n > 0:
        dst_old = slice_view(result, [slice(0, old_n), slice(None)])
        assign_tensor(dst_old, experience_tensor)
        result.data[:old_n] = experience_tensor.data

    # Copy new row into last slot
    dst_new = slice_view(result, [slice(old_n, new_n), slice(None)])
    assign_tensor(dst_new, new_row)
    result.data[old_n] = new_row.data[0]

    return result
