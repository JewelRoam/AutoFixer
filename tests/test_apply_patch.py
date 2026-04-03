"""Tests for autofixer.optim.apply_patch — experience write-back and workspace patching.

Tests that:
1. A DiffPatch can be applied to a file on disk
2. Experience can be written (appended) to the experience tensor
3. The backward-then-patch lifecycle works end-to-end
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


class TestWriteExperience:
    def test_append_experience_to_tensor(self):
        from autofixer.optim.apply_patch import append_experience
        from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

        with tempfile.TemporaryDirectory() as tmpdir:
            # Start with 1-entry experience
            initial = make_tensor(
                [["query_kw", "old key text", "old value text"]],
                tmpdir,
            )
            snapshot = ContextSnapshot(
                exception_type="KeyError",
                traceback="...",
                target_file="/project/app.py",
                crash_line_num=5,
                source_code="d = {}\nv = d['missing']",
                local_vars='{"d": {}}',
            )
            diff_patch = "--- a/app.py\n+++ b/app.py\n@@ -1 +1,2 @@\n d = {}\n+v = d.get('missing', None)\n"

            new_exp = append_experience(
                experience_tensor=initial,
                snapshot=snapshot,
                diff_patch=diff_patch,
                relative_to=tmpdir,
            )
            # Should have 2 rows now
            assert new_exp.shape[0] == initial.shape[0] + 1
            assert new_exp.shape[1] == 3


class TestExperienceLifecycle:
    def test_write_then_read_back(self):
        """Verify that written experience can be read back correctly."""
        from autofixer.optim.apply_patch import append_experience
        from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
        from experience.symbolic_tensor.tensor_util.pack_tensor import pack_tensor

        with tempfile.TemporaryDirectory() as tmpdir:
            initial = make_tensor(
                [["initial_query", "initial_key", "initial_value"]],
                tmpdir,
            )
            snapshot = ContextSnapshot(
                exception_type="TypeError",
                traceback="...",
                target_file="/a.py",
                crash_line_num=1,
                source_code="x = 1 + 'a'",
                local_vars='{"x": 1}',
            )
            diff = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-x = 1 + 'a'\n+x = 1 + str('a')\n"

            new_exp = append_experience(initial, snapshot, diff, tmpdir)
            packed = pack_tensor(new_exp)
            assert "initial_key" in packed
            assert "TypeError" in packed or "x = 1" in packed
