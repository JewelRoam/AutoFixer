#!/usr/bin/env python3
"""End-to-end integration test: forward → compute_loss → backward → step.

Requires LLM credentials configured in ~/.anthropic.sh or environment.
Run:
    PYTHONPATH=.:experience python scripts/test_e2e_autograd.py
"""

import os
import sys
import subprocess
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experience"))

# Load LLM env
sh_path = os.path.expanduser("~/.anthropic.sh")
if os.path.isfile(sh_path):
    result = subprocess.run(
        ["bash", "-c", f"source {sh_path} && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            if key.startswith("LLM_"):
                os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

import torch
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from autofixer.model.bugfix_agent import AutoFixerAgent, read_tensor_element


def read_storage(tensor, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    if os.path.isfile(os.path.realpath(path)):
        with open(os.path.realpath(path)) as f:
            return f.read()
    return "(empty)"


def main():
    print("=" * 60)
    print("E2E Autograd Test: forward → compute_loss → backward → step")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── 1. Build agent with seed experience ──
        print("\n[1] Building agent with 4 slots, seeding 1 experience...")
        agent = AutoFixerAgent(num_experience_slots=4, topk=1)
        optimizer = StSGD(agent.parameters(), lr=1.0)

        # Seed one experience row
        seed_exp = make_tensor([
            [
                "ZeroDivisionError\ndivide\nreturn",
                "## Exception: ZeroDivisionError\ndef divide(a, b):\n    return a / b\nlocal_vars: {\"a\": 10, \"b\": 0}",
                "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,4 @@\n def divide(a, b):\n+    if b == 0:\n+        raise ValueError('cannot divide by zero')\n     return a / b\n",
            ],
            ["", "", ""],  # empty slot 1
            ["", "", ""],  # empty slot 2
            ["", "", ""],  # empty slot 3
        ], tmpdir)

        from experience.symbolic_tensor.function.st_copy import copy_impl
        loaded = copy_impl(seed_exp, agent.moe._experience_dir)
        agent.moe.experience = loaded
        agent.moe.experience.requires_grad_(True)

        exp = agent.get_experience()
        print(f"  Experience shape: {list(exp.shape)}")
        print(f"  Slot 0 query: {read_storage(exp, 0)[:80]}...")

        # ── 2. Forward pass ──
        print("\n[2] Forward pass (LLM call)...")
        input_text = (
            "## Exception: ZeroDivisionError\n"
            "## File: /project/math_utils.py:5\n\n"
            "### Source Code\n"
            "def safe_divide(x, y):\n"
            "    result = x / y  # BUG: no zero guard\n"
            "    return result\n\n"
            "### Local Variables\n"
            '{"x": 42, "y": 0}\n'
        )
        input_tensor = make_tensor([input_text], tmpdir)
        input_tensor.requires_grad_(True)

        output, selected_indexes = agent(input_tensor)
        raw_output = read_tensor_element(output, 0)

        print(f"  Selected indexes: {selected_indexes}")
        print(f"  Output ({len(raw_output)} chars):")
        for line in raw_output.splitlines()[:10]:
            print(f"    {line}")
        if len(raw_output.splitlines()) > 10:
            print(f"    ... ({len(raw_output.splitlines())} lines)")

        # ── 3. Compute loss ──
        print("\n[3] Computing loss...")
        correct_patch = (
            "--- a/math_utils.py\n"
            "+++ b/math_utils.py\n"
            "@@ -1,3 +1,5 @@\n"
            " def safe_divide(x, y):\n"
            "+    if y == 0:\n"
            "+        raise ValueError('division by zero')\n"
            "     result = x / y\n"
            "     return result\n"
        )
        expected_tensor = make_tensor([correct_patch], tmpdir)

        optimizer.zero_grad()
        loss = agent.compute_loss(output, expected_tensor)
        loss_val = loss.item()
        print(f"  Loss = {loss_val:.4f}")

        if loss_val == 0.0:
            print("  Loss is 0 — LLM output matched expected exactly! (rare but great)")
            print("  Skipping backward (no gradient needed).")
            return

        # ── 4. Backward ──
        print("\n[4] loss.backward() (symbolic gradient propagation)...")
        loss.backward()

        exp = agent.get_experience()
        grad = exp.grad
        if grad is not None:
            print(f"  Grad shape: {list(grad.shape)}")
            print(f"  Grad nonzero: {torch.count_nonzero(grad.data).item()} elements")
            # Show grad content if st_* attributes are available
            if hasattr(grad, 'st_relative_to'):
                for i in range(grad.numel()):
                    gtext = read_storage(grad, i)
                    if gtext.strip() and gtext.strip() != "TODO":
                        print(f"  Grad[{i}]: {gtext[:80]}...")
            else:
                print("  (grad tensor has no st_* attrs — symbolic metadata is in symbolic_grad_registry)")
        else:
            print("  WARNING: grad is None (symbolic_grad_registry may have been used)")
            # The grad may be in the registry instead — optimizer.step() handles this
            print("  (This is expected — StSGD retrieves it from symbolic_grad_registry)")

        # ── 5. Optimizer step ──
        print("\n[5] optimizer.step() (patching experience tensor)...")
        optimizer.step()
        stats = optimizer.get_last_step_stats()
        print(f"  Stats: {stats}")

        # ── 6. Verify experience was updated ──
        print("\n[6] Verifying experience after update...")
        exp = agent.get_experience()
        for row in range(exp.shape[0]):
            q = read_storage(exp, row * 3 + 0)
            k = read_storage(exp, row * 3 + 1)
            v = read_storage(exp, row * 3 + 2)
            has_content = bool(q.strip()) or bool(k.strip()) or bool(v.strip())
            if has_content:
                print(f"  Row {row}: query={len(q)}c key={len(k)}c value={len(v)}c")
                print(f"    query[:60]: {q[:60]}")
            else:
                print(f"  Row {row}: (empty)")

    print("\n" + "=" * 60)
    print("E2E test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
