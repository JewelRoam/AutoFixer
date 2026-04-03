# AutoFixer

Experience-based bug-fix agent built on the [Experience](https://github.com/lixinqi/Experience) symbolic tensor framework.

AutoFixer intercepts Python exceptions at runtime, retrieves similar historical bug-fix experiences via Jaccard similarity, and generates unified diff patches through an LLM. Successful fixes are sedimented back into the experience tensor, enabling continuous learning.

## Architecture

```
ContextSnapshot ──► AutoFixerAgent (StMoeModule) ──► DiffPatch
                        │
                  Experience Tensor [N, 3]
                  columns = [query, key, value]
```

**4-Phase Lifecycle:**

| Phase | Description |
|-------|-------------|
| Phase 1 — Distill | Mine bug-fix commits from git history into experience rows |
| Phase 2 — Forward | Intercept exception → build ContextSnapshot → retrieve + generate patch |
| Phase 3 — Evaluate | Apply patch to workspace, developer confirms result |
| Phase 4 — Backward | On failure, collect correct diff → `compute_loss` → `loss.backward()` → `optimizer.step()` updates experience via autograd |

**Autograd pipeline (Phase 4 detail):**

```python
output, indexes = agent(input_tensor)               # forward
loss = agent.compute_loss(output, expected_tensor)   # edit-distance loss
loss.backward()                                      # symbolic gradients → experience
optimizer.step()                                     # StSGD patches experience tensor
```

Loss is computed via `get_edit_distance_ratio` (Levenshtein distance). Gradients are symbolic text diffs propagated through `StMoe.backward`. `StSGD` patches the experience tensor in-place — empty slots are auto-filled via cold-start.

## Setup

### Prerequisites

- Python >= 3.10
- Git

### Install

```bash
git clone --recurse-submodules https://github.com/JewelRoam/AutoFixer.git
cd AutoFixer

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Configure LLM

The Experience framework uses an **OpenAI-compatible API** with three environment variables:

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key for the LLM endpoint |
| `LLM_BASE_URL` | Base URL (e.g. `https://api.openai.com/v1`) |
| `LLM_MODEL` | Model identifier (e.g. `gpt-4`, `claude-sonnet-4-20250514`) |

The framework convention is to store these in `~/.llm_config.sh`:

```bash
# ~/.llm_config.sh
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4"
```

The REPL script automatically sources this file on startup. Alternatively, you can export the variables directly in your shell or pass `--llm-model` to override the model name.

## Usage

### Run Tests

```bash
# Unit tests
PYTHONPATH=.:experience pytest tests/ -v

# End-to-end autograd pipeline test (requires LLM credentials)
PYTHONPATH=.:experience python scripts/test_e2e_autograd.py
```

The e2e test validates the full lifecycle: build agent → seed experience → forward (LLM call) → compute loss → backward (symbolic gradient propagation) → optimizer step → verify experience update.

### Interactive REPL

```bash
PYTHONPATH=.:experience python scripts/run_repl.py
```

**REPL commands:** (Tab completes both commands and file paths)

| Command | Description |
|---------|-------------|
| `hook <script.py>` | Run a Python script with exception interception. When the script crashes, AutoFixer captures the full context and enters the fix cycle. |
| `snap <context.json>` | Load crash context from a JSON file (see `scripts/demo_context.json` for format). |
| `distill <repo_path>` | Mine bug-fix history from a git repo into experience tensor. |
| `status` | Show agent experience shape, storage path, and LLM config. |
| `quit` | Exit the REPL. |

**CLI options:**

```
--distill PATH       Distill bug-fix history from a git repo (Phase 1 cold-start)
--llm-model MODEL    Override LLM model name (e.g. gpt-4)
--num-experience N   Number of experience slots (default: 32)
--topk K             Top-k experience retrieval (default: 3)
--lr LR              Learning rate for StSGD optimizer (default: 1.0)
```

### Example: Fix a Buggy Script

```bash
# 1. Start the REPL
PYTHONPATH=.:experience python scripts/run_repl.py

# 2. Use Tab completion to locate and run the demo script
autofixer> hook scripts/demo_buggy.py
#          ^^^^  ^^^^^^^ Tab completes file paths

# 3. AutoFixer captures the ZeroDivisionError, retrieves similar
#    experiences, and generates a patch via LLM (Phase 2)
# 4. Review and apply the patch (Phase 3)
# 5. If the fix fails, provide the correct diff to train (Phase 4)
```

### Example: Load Crash Context from JSON

```bash
# Use snap with a pre-built context file
autofixer> snap scripts/demo_context.json
```

The JSON format matches ContextSnapshot fields:

```json
{
  "exception_type": "ZeroDivisionError",
  "traceback": "...",
  "target_file": "scripts/demo_buggy.py",
  "crash_line_num": 8,
  "source_code": "...",
  "local_vars": "{\"count\": 0}"
}
```

### Example: Cold-Start from Git History

```bash
PYTHONPATH=.:experience python scripts/run_repl.py --distill /path/to/your/repo
```

This scans the repo for commits containing fix/bug/patch keywords and distills them into experience rows before entering the interactive REPL.

## Project Structure

```
autofixer/
    context_snapshot.py      # ContextSnapshot dataclass (Viba schema)
    env_interceptor/
        frame_inspector.py   # Extract crash context from exception frames
        sys_excepthook.py    # Global exception hook (install/uninstall)
    model/
        bugfix_agent.py      # AutoFixerAgent wrapping StMoeModule
    optim/
        apply_patch.py       # Apply unified diff to workspace files
    tools/
        git_miner.py         # Git history distillation
scripts/
    run_repl.py              # Interactive REPL entry point
    test_e2e_autograd.py     # E2E autograd pipeline smoke test
    demo_buggy.py            # Demo script with a ZeroDivisionError bug
    demo_context.json        # Demo crash context for snap command
experience/                  # git submodule: lixinqi/Experience
tests/                       # Unit tests (pytest)
```

## License

See [LICENSE](LICENSE).
