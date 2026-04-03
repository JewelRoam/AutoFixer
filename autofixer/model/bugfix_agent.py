"""AutoFixerAgent: nn.Module wrapping Experience StMoeModule for bug fixing.

This module converts ContextSnapshot into symbolic tensors, delegates to
the StMoe mixed-expert layer for experience-based retrieval + LLM generation,
and outputs DiffPatch text.

Autograd lifecycle:
    output, indexes = agent(input_tensor)
    expected = make_tensor([correct_patch], tmpdir)
    loss = agent.compute_loss(output, expected)
    loss.backward()       # symbolic gradients flow through StMoe → experience
    optimizer.step()      # StSGD patches experience tensor in-place
    optimizer.zero_grad()
"""

import os
import re
import tempfile
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from experience.symbolic_tensor.module.st_moe import StMoeModule
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_edit_distance_ratio import (
    get_edit_distance_ratio,
)

from autofixer.context_snapshot import ContextSnapshot


# ── Utility: read text content from a symbolic tensor element ──

def read_tensor_element(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the text content stored at a flat index in a symbolic tensor."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# ── Utility: convert ContextSnapshot → symbolic tensor ──

def snapshot_to_tensor(
    snapshot: ContextSnapshot,
    relative_to: str,
) -> torch.Tensor:
    """Convert a ContextSnapshot into a [1]-shaped symbolic tensor.

    The tensor element contains the full crash context text including
    both source_code and local_vars (the two mandatory anchor inputs).
    """
    text = snapshot.to_text()
    return make_tensor([text], relative_to)


# ── Utility: parse DiffPatch from raw LLM output ──

_DIFF_HEADER_RE = re.compile(r"^---\s", re.MULTILINE)


def parse_diff_patch(raw: str) -> Optional[str]:
    """Extract a valid unified diff from raw text.

    Returns the diff string if it looks like a valid unified diff,
    or None if the input doesn't contain a recognizable patch.
    """
    if not raw or not raw.strip():
        return None
    if _DIFF_HEADER_RE.search(raw) and "+++" in raw:
        return raw.strip()
    return None


# ── The Agent Model ──

_TASK_PROMPT = """\
You are an expert bug-fixing agent. Given the crash context (exception type, \
traceback, source code, and local variables) along with similar historical \
bug-fix experiences, generate a unified diff patch (diff -u format) that fixes \
the bug. Output ONLY the patch, no explanation."""


class AutoFixerAgent(nn.Module):
    """Experience-based bug fix agent built on StMoeModule.

    The experience tensor is pre-allocated at `[num_experience_slots, 3]` with
    empty-string storage.  StSGD's cold-start mechanism auto-fills empty slots
    during `optimizer.step()` — no manual append needed.

    Args:
        num_experience_slots: Number of experience entries to pre-allocate.
        topk: Number of experiences to retrieve per input.
        task_prompt: Override the default task prompt for the LLM.
        llm_env: Environment variables for the LLM client.
    """

    def __init__(
        self,
        num_experience_slots: int = 16,
        topk: int = 3,
        task_prompt: str = _TASK_PROMPT,
        llm_env: Optional[dict] = None,
    ):
        super().__init__()
        self.num_experience_slots = num_experience_slots
        self.topk = topk
        self.moe = StMoeModule(
            experience_shape=[num_experience_slots, 3],
            task_prompt=task_prompt,
            topk=topk,
            llm_env=llm_env,
        )

    def get_experience(self) -> torch.Tensor:
        """Return the underlying experience tensor."""
        return self.moe.experience

    def parameters(self, recurse: bool = True):
        """Yield model parameters (the experience tensor)."""
        return self.moe.parameters(recurse)

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Forward pass: input symbolic tensor → output DiffPatch tensor.

        Args:
            input_tensor: A symbolic tensor of shape [N] containing crash context text.

        Returns:
            (output_tensor, selected_indexes): output contains generated patches,
            selected_indexes tracks which experience entries were used.
        """
        return self.moe(input_tensor)

    @staticmethod
    def compute_loss(
        output: torch.Tensor, expected: torch.Tensor
    ) -> torch.Tensor:
        """Compute symbolic edit-distance loss between output and expected tensors.

        This is the entry point to the autograd backward graph.  Calling
        `loss.backward()` will propagate symbolic diff-gradients through
        StMoe and into the experience tensor.

        Args:
            output: Symbolic tensor from forward() containing generated patches.
            expected: Symbolic tensor containing the correct/human-verified patches.

        Returns:
            Scalar loss tensor (mean edit-distance ratio).
        """
        ratios = get_edit_distance_ratio(output, expected)
        return ratios.mean()
