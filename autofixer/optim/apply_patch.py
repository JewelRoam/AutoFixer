"""Apply patch to workspace.

Handles applying a unified diff patch to a real file on disk.

Note: experience append is now handled by the autograd pipeline:
    loss = agent.compute_loss(output, expected)
    loss.backward()
    optimizer.step()  # StSGD auto-patches experience via cold-start
"""

import os
import subprocess
import tempfile


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
