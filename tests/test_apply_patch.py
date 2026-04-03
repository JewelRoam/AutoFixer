"""Tests for autofixer.optim.apply_patch — workspace patching.

Tests that:
1. A DiffPatch can be applied to a file on disk
2. The autograd backward lifecycle (compute_loss → backward → step) works
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experience"))

from autofixer.context_snapshot import ContextSnapshot


@pytest.fixture
def buggy_file(tmp_path):
    """Create a buggy Python file for patching tests."""
    f = tmp_path / "buggy.py"
    f.write_text("def divide(a, b):\n    return a / b\n")
    return f


@pytest.fixture
def fix_patch():
    """A valid unified diff patch."""
    return (
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def divide(a, b):\n"
        "+    if b == 0:\n"
        "+        raise ValueError('division by zero')\n"
        "     return a / b\n"
    )


class TestApplyPatchToWorkspace:
    def test_apply_patch_modifies_file(self, buggy_file, fix_patch):
        from autofixer.optim.apply_patch import apply_patch_to_file
        success = apply_patch_to_file(str(buggy_file), fix_patch)
        assert success is True
        content = buggy_file.read_text()
        assert "if b == 0:" in content
        assert "raise ValueError" in content

    def test_apply_patch_returns_false_on_bad_patch(self, buggy_file):
        from autofixer.optim.apply_patch import apply_patch_to_file
        bad_patch = "--- a/wrong.py\n+++ b/wrong.py\n@@ -99,1 +99,1 @@\n-nonexistent line\n+replaced\n"
        success = apply_patch_to_file(str(buggy_file), bad_patch)
        assert success is False


class TestAutogradLifecycle:
    """Verify that compute_loss → backward produces gradients on experience."""

    def test_compute_loss_produces_grad(self):
        from autofixer.model.bugfix_agent import AutoFixerAgent
        from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = os.path.realpath(raw_tmpdir)
            output = make_tensor(["wrong answer"], tmpdir)
            output.requires_grad_(True)
            expected = make_tensor(["correct answer"], tmpdir)

            loss = AutoFixerAgent.compute_loss(output, expected)
            assert loss.item() > 0.0
            assert loss.requires_grad

    def test_optimizer_step_updates_experience(self):
        """Verify StSGD.step() can process symbolic gradients."""
        from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
        from experience.symbolic_tensor.optimizer.st_sgd import StSGD

        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = os.path.realpath(raw_tmpdir)
            # Create a small experience tensor
            exp = make_tensor([["query", "old key", "old value"]], tmpdir)
            exp.requires_grad_(True)

            optimizer = StSGD([exp], lr=1.0)

            # Manually create a diff gradient (simulates backward output)
            diff_key = (
                "--- data\n+++ data\n@@ -1 +1 @@\n"
                "-old key\n+new key\n"
            )
            diff_value = (
                "--- data\n+++ data\n@@ -1 +1 @@\n"
                "-old value\n+new value\n"
            )
            grad = make_tensor([["", diff_key, diff_value]], tmpdir)
            grad.data.fill_(0.0)
            grad.data[0, 1] = 1.0
            grad.data[0, 2] = 1.0
            exp.grad = grad

            optimizer.step()
            stats = optimizer.get_last_step_stats()
            assert stats["applied"] == 2
