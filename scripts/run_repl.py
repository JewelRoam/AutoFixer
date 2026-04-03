#!/usr/bin/env python3
"""AutoFixer REPL: Interactive bug-fix agent with experience-based learning.

This script implements the full 4-phase lifecycle:
  Phase 1: Git history distillation (offline, optional)
  Phase 2: Forward pass — exception interception → diagnosis → patch generation
  Phase 3: Evaluation — apply patch, run tests or ask developer
  Phase 4: Backward — experience sedimentation on failure

Usage:
    # Start interactive REPL
    python scripts/run_repl.py

    # Distill git history first (cold-start)
    python scripts/run_repl.py --distill /path/to/repo
"""

import argparse
import os
import runpy
import subprocess
import sys
import tempfile

# Ensure project root and experience submodule are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experience"))

import torch

from autofixer.context_snapshot import ContextSnapshot
from autofixer.model.bugfix_agent import (
    AutoFixerAgent,
    snapshot_to_tensor,
    read_tensor_element,
    parse_diff_patch,
)
from autofixer.optim.apply_patch import apply_patch_to_file, append_experience
from autofixer.tools.git_miner import distill_repo_to_experience_rows
from autofixer.env_interceptor.sys_excepthook import (
    install_hook,
    uninstall_hook,
    get_last_snapshot,
)
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.optimizer.st_sgd import StSGD

# Persistent experience storage
DATA_DIR = os.path.realpath(os.path.join(PROJECT_ROOT, "data", "shared_experience"))
os.makedirs(DATA_DIR, exist_ok=True)


def _load_llm_env_from_shell() -> None:
    """Source ~/.anthropic.sh and inject LLM_* vars into os.environ.

    This follows the Experience framework convention: developers store
    LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in ~/.anthropic.sh.
    """
    sh_path = os.path.expanduser("~/.anthropic.sh")
    if not os.path.isfile(sh_path):
        return
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


def phase1_distill(repo_path: str, agent: AutoFixerAgent) -> None:
    """Phase 1: Distill git history into experience tensor (cold-start)."""
    print(f"[Phase 1] Distilling bug-fix history from {repo_path}...")
    rows = distill_repo_to_experience_rows(repo_path)
    if not rows:
        print("  No bug-fix commits found.")
        return

    print(f"  Found {len(rows)} bug-fix experience(s). Loading into agent...")
    experience = make_tensor(rows, DATA_DIR)
    # Replace agent's experience with distilled data
    from experience.symbolic_tensor.function.st_copy import copy_impl
    loaded = copy_impl(experience, agent.moe._experience_dir)
    agent.moe.experience = loaded
    agent.moe.experience.requires_grad_(True)
    print(f"  Experience loaded: shape {list(agent.moe.experience.shape)}")


def phase2_forward(snapshot: ContextSnapshot, agent: AutoFixerAgent) -> str:
    """Phase 2: Forward pass — generate fix patch from crash context."""
    print("[Phase 2] Running forward pass (experience retrieval + LLM generation)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = os.path.realpath(tmpdir)
        input_tensor = snapshot_to_tensor(snapshot, relative_to=tmpdir)
        output, selected_indexes = agent(input_tensor)
        raw_output = read_tensor_element(output, 0)
        print(f"  Raw output ({len(raw_output)} chars)")
        return raw_output


def phase3_evaluate(
    patch_text: str, target_file: str
) -> bool:
    """Phase 3: Apply patch and let developer evaluate."""
    patch = parse_diff_patch(patch_text)
    if patch is None:
        print("[Phase 3] No valid patch found in output.")
        return False

    print(f"[Phase 3] Generated patch:\n{patch[:500]}{'...' if len(patch) > 500 else ''}")
    response = input("\nApply this patch? [y/n/m(manual)]: ").strip().lower()

    if response == "y":
        success = apply_patch_to_file(target_file, patch)
        if success:
            print("  Patch applied successfully.")
            test_result = input("Did the fix work? [y/n]: ").strip().lower()
            return test_result == "y"
        else:
            print("  Patch application failed.")
            return False
    elif response == "m":
        print("  Entering manual fix mode. Fix the bug yourself, then press Enter.")
        input("  Press Enter when done...")
        return False  # triggers Phase 4 backward
    else:
        return False


def phase4_backward(
    snapshot: ContextSnapshot,
    agent: AutoFixerAgent,
    optimizer: StSGD,
) -> None:
    """Phase 4: Backward pass — sediment experience from developer's fix."""
    print("[Phase 4] Recording experience from this fix attempt...")
    correct_diff = input(
        "Paste the correct diff (end with EOF on a new line), or 'skip' to skip:\n"
    )
    if correct_diff.strip().lower() == "skip":
        print("  Skipped experience recording.")
        return

    # Collect until EOF
    lines = [correct_diff]
    while True:
        line = input()
        if line.strip() == "EOF":
            break
        lines.append(line)
    correct_diff = "\n".join(lines)

    new_exp = append_experience(
        agent.moe.experience, snapshot, correct_diff, DATA_DIR
    )
    agent.moe.experience = new_exp
    agent.moe.experience.requires_grad_(True)
    print(f"  Experience updated: shape {list(agent.moe.experience.shape)}")


def interactive_repl(agent: AutoFixerAgent, optimizer: StSGD) -> None:
    """Main REPL loop for interactive bug fixing."""
    print("\n=== AutoFixer REPL ===")
    print("Commands:")
    print("  hook   — Install exception hook on a Python script")
    print("  snap   — Manually input a crash context")
    print("  quit   — Exit")

    while True:
        cmd = input("\nautofixer> ").strip().lower()

        if cmd == "quit":
            break
        elif cmd == "hook":
            script = input("Python script to run: ").strip()
            if not os.path.isfile(script):
                print(f"  File not found: {script}")
                continue
            install_hook()
            try:
                runpy.run_path(script, run_name="__main__")
            except SystemExit:
                pass
            except Exception:
                pass  # excepthook already captured the snapshot
            finally:
                uninstall_hook()

            snapshot = get_last_snapshot()
            if snapshot is None:
                print("  No exception captured.")
                continue
            _handle_snapshot(snapshot, agent, optimizer)

        elif cmd == "snap":
            print("Enter crash details:")
            exc_type = input("  Exception type: ").strip()
            tb = input("  Traceback (one line): ").strip()
            target = input("  Target file: ").strip()
            line_num = int(input("  Crash line number: ").strip())
            source = input("  Source code snippet: ").strip()
            local_vars = input("  Local vars (JSON): ").strip()
            snapshot = ContextSnapshot(
                exception_type=exc_type,
                traceback=tb,
                target_file=target,
                crash_line_num=line_num,
                source_code=source,
                local_vars=local_vars,
            )
            _handle_snapshot(snapshot, agent, optimizer)
        else:
            print(f"  Unknown command: {cmd}")


def _handle_snapshot(
    snapshot: ContextSnapshot,
    agent: AutoFixerAgent,
    optimizer: StSGD,
) -> None:
    """Process a single crash snapshot through phases 2-4."""
    print(f"\n  Exception: {snapshot.exception_type}")
    print(f"  File: {snapshot.target_file}:{snapshot.crash_line_num}")

    raw_output = phase2_forward(snapshot, agent)
    success = phase3_evaluate(raw_output, snapshot.target_file)

    if success:
        print("\n[Result] Fix successful! No backward pass needed (loss=0).")
    else:
        phase4_backward(snapshot, agent, optimizer)


def main():
    parser = argparse.ArgumentParser(description="AutoFixer: Experience-based bug fix agent")
    parser.add_argument(
        "--distill", type=str, default=None,
        help="Path to git repo to distill bug-fix history from (Phase 1)",
    )
    parser.add_argument(
        "--num-experience", type=int, default=32,
        help="Number of experience slots (default: 32)",
    )
    parser.add_argument("--topk", type=int, default=3, help="Top-k experience retrieval")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate for StSGD")
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="LLM model name (overrides LLM_MODEL env var)",
    )
    args = parser.parse_args()

    # Load LLM credentials from ~/.anthropic.sh (Experience convention)
    _load_llm_env_from_shell()

    # Build llm_env dict if model is specified
    llm_env = None
    if args.llm_model is not None:
        llm_env = {
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
            "LLM_BASE_URL": os.environ.get("LLM_BASE_URL", ""),
            "LLM_MODEL": args.llm_model,
        }

    print("Initializing AutoFixerAgent...")
    agent = AutoFixerAgent(
        num_experience_slots=args.num_experience,
        topk=args.topk,
        llm_env=llm_env,
    )
    optimizer = StSGD(agent.parameters(), lr=args.lr)

    if args.distill:
        phase1_distill(args.distill, agent)

    interactive_repl(agent, optimizer)
    print("Goodbye.")


if __name__ == "__main__":
    main()
