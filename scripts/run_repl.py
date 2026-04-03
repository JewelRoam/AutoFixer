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
import glob as globmod
import json
import os
import readline
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

# ── readline Tab-completion ────────────────────────────────────────────

_COMMANDS = ["hook", "snap", "distill", "status", "quit"]


class _Completer:
    """Tab completer: commands for first token, file paths for arguments."""

    def __init__(self):
        self._matches = []

    def complete(self, text, state):
        if state == 0:
            line = readline.get_line_buffer().lstrip()
            begin = readline.get_begidx()

            if begin == 0 or " " not in line:
                # Completing the command name
                self._matches = [c + " " for c in _COMMANDS if c.startswith(text)]
            else:
                # Completing a file path argument
                self._matches = []
                for path in globmod.glob(text + "*"):
                    if os.path.isdir(path):
                        self._matches.append(path + os.sep)
                    else:
                        self._matches.append(path + " ")
        if state < len(self._matches):
            return self._matches[state]
        return None


def _setup_readline():
    """Configure readline with tab completion and history."""
    comp = _Completer()
    readline.set_completer(comp.complete)
    # macOS uses libedit which needs a different syntax
    if "libedit" in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    # Treat / as part of the word so file paths complete correctly
    readline.set_completer_delims(" \t\n;")


# ── LLM environment ───────────────────────────────────────────────────

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


# ── Phase implementations ─────────────────────────────────────────────

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
    print("  Paste the correct diff (end with EOF on a new line), or 'skip':")
    first_line = input("  ").strip()
    if first_line.lower() == "skip":
        print("  Skipped experience recording.")
        return

    # Collect until EOF
    lines = [first_line]
    while True:
        line = input("  ")
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


# ── Command handlers ──────────────────────────────────────────────────

def _cmd_hook(arg: str, agent: AutoFixerAgent, optimizer: StSGD) -> None:
    """hook <script.py> — Run a script and intercept exceptions."""
    script = arg.strip()
    if not script:
        print("  Usage: hook <script.py>")
        return
    if not os.path.isfile(script):
        print(f"  File not found: {script}")
        return

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
        return
    _handle_snapshot(snapshot, agent, optimizer)


def _cmd_snap(arg: str, agent: AutoFixerAgent, optimizer: StSGD) -> None:
    """snap <context.json> — Load crash context from a JSON file."""
    path = arg.strip()
    if not path:
        print("  Usage: snap <context.json>")
        print("  JSON fields: exception_type, traceback, target_file,")
        print("               crash_line_num, source_code, local_vars")
        return
    if not os.path.isfile(path):
        print(f"  File not found: {path}")
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            snapshot = ContextSnapshot.from_json(f.read())
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  Invalid context JSON: {e}")
        return
    _handle_snapshot(snapshot, agent, optimizer)


def _cmd_distill(arg: str, agent: AutoFixerAgent) -> None:
    """distill <repo_path> — Mine bug-fix history from a git repo."""
    repo = arg.strip()
    if not repo:
        print("  Usage: distill <repo_path>")
        return
    if not os.path.isdir(repo):
        print(f"  Not a directory: {repo}")
        return
    phase1_distill(repo, agent)


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


# ── REPL ──────────────────────────────────────────────────────────────

def interactive_repl(agent: AutoFixerAgent, optimizer: StSGD) -> None:
    """Main REPL loop for interactive bug fixing."""
    _setup_readline()

    print("\n=== AutoFixer REPL ===")
    print("Commands:  (Tab completes commands and file paths)")
    print("  hook <script.py>     Run script, intercept exceptions")
    print("  snap <context.json>  Load crash context from JSON file")
    print("  distill <repo_path>  Mine bug-fix history from a git repo")
    print("  status               Show agent experience info")
    print("  quit                 Exit")

    while True:
        try:
            line = input("\nautofixer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "quit":
            break
        elif cmd == "hook":
            _cmd_hook(arg, agent, optimizer)
        elif cmd == "snap":
            _cmd_snap(arg, agent, optimizer)
        elif cmd == "distill":
            _cmd_distill(arg, agent)
        elif cmd == "status":
            exp = agent.get_experience()
            print(f"  Experience shape: {list(exp.shape)}")
            print(f"  Storage: {DATA_DIR}")
            llm_key = os.environ.get("LLM_API_KEY", "")
            llm_model = os.environ.get("LLM_MODEL", "(not set)")
            print(f"  LLM model: {llm_model}")
            print(f"  LLM API key: {'***' + llm_key[-4:] if len(llm_key) > 4 else '(not set)'}")
        else:
            print(f"  Unknown command: {cmd}")
            print("  Type a command or press Tab for completion.")


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
