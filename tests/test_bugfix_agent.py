"""Tests for autofixer.model.bugfix_agent — AutoFixerAgent nn.Module.

Tests that the agent:
1. Properly wraps Experience StMoeModule
2. Converts ContextSnapshot to symbolic tensor input
3. Produces DiffPatch output from forward pass
4. Integrates with StSGD optimizer
"""

import os
import sys
import tempfile
import pytest
import torch

# Add experience to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experience"))

from autofixer.context_snapshot import ContextSnapshot


@pytest.fixture
def sample_snapshot():
    return ContextSnapshot(
        exception_type="ZeroDivisionError",
        traceback="Traceback:\n  File \"calc.py\", line 10\nZeroDivisionError: division by zero",
        target_file="/project/calc.py",
        crash_line_num=10,
        source_code="def divide(a, b):\n    return a / b  # BUG: no zero check\n",
        local_vars='{"a": 10, "b": 0}',
    )


class TestAutoFixerAgentInit:
    def test_construction_with_defaults(self):
        from autofixer.model.bugfix_agent import AutoFixerAgent
        agent = AutoFixerAgent(num_experience_slots=4)
        assert agent.num_experience_slots == 4
        assert agent.topk >= 1

    def test_experience_tensor_shape(self):
        from autofixer.model.bugfix_agent import AutoFixerAgent
        agent = AutoFixerAgent(num_experience_slots=8, topk=2)
        # Experience should be [N, 3] where 3 = (query, key, value)
        exp = agent.get_experience()
        assert len(exp.shape) == 2
        assert exp.shape[0] == 8
        assert exp.shape[1] == 3

    def test_experience_requires_grad(self):
        from autofixer.model.bugfix_agent import AutoFixerAgent
        agent = AutoFixerAgent(num_experience_slots=4)
        assert agent.get_experience().requires_grad is True

    def test_parameters_yields_experience(self):
        from autofixer.model.bugfix_agent import AutoFixerAgent
        agent = AutoFixerAgent(num_experience_slots=4)
        params = list(agent.parameters())
        assert len(params) == 1


class TestSnapshotToTensor:
    def test_snapshot_to_tensor_returns_symbolic_tensor(self, sample_snapshot):
        from autofixer.model.bugfix_agent import snapshot_to_tensor
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = snapshot_to_tensor(sample_snapshot, relative_to=tmpdir)
            assert hasattr(tensor, "st_tensor_uid")
            assert hasattr(tensor, "st_relative_to")
            assert tensor.shape == (1,)  # single input element

    def test_snapshot_tensor_content_includes_source_and_vars(self, sample_snapshot):
        from autofixer.model.bugfix_agent import snapshot_to_tensor, read_tensor_element
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = snapshot_to_tensor(sample_snapshot, relative_to=tmpdir)
            content = read_tensor_element(tensor, 0)
            # Must contain both source_code and local_vars (per architecture spec)
            assert "a / b" in content
            assert '"b": 0' in content


class TestDiffPatchOutput:
    def test_parse_diff_patch(self):
        from autofixer.model.bugfix_agent import parse_diff_patch
        raw = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,3 @@\n def divide(a, b):\n+    if b == 0: return None\n     return a / b\n"
        patch = parse_diff_patch(raw)
        assert patch is not None
        assert "---" in patch
        assert "+++ " in patch

    def test_parse_diff_patch_empty_returns_none(self):
        from autofixer.model.bugfix_agent import parse_diff_patch
        assert parse_diff_patch("") is None
        assert parse_diff_patch("no diff here") is None


class TestAgentWithOptimizer:
    def test_compatible_with_stsgd(self):
        from autofixer.model.bugfix_agent import AutoFixerAgent
        from experience.symbolic_tensor.optimizer.st_sgd import StSGD
        agent = AutoFixerAgent(num_experience_slots=4)
        optimizer = StSGD(agent.parameters(), lr=1.0)
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]["params"]) == 1
