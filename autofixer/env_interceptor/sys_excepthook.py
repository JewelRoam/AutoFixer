"""Global exception hook: intercept unhandled exceptions and build ContextSnapshot.

Usage:
    from autofixer.env_interceptor.sys_excepthook import install_hook, uninstall_hook, get_last_snapshot
    install_hook()
    # ... run code that may raise ...
    snapshot = get_last_snapshot()  # ContextSnapshot or None
    uninstall_hook()
"""

import sys
from types import TracebackType
from typing import Optional, Type

from autofixer.context_snapshot import ContextSnapshot
from autofixer.env_interceptor.frame_inspector import build_snapshot_from_exception

_original_hook = None
_last_snapshot: Optional[ContextSnapshot] = None


def _autofixer_excepthook(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    """Custom excepthook that captures a ContextSnapshot before delegating."""
    global _last_snapshot
    _last_snapshot = build_snapshot_from_exception(exc_type, exc_value, exc_tb)
    # Still call the original hook so tracebacks are printed
    if _original_hook is not None:
        _original_hook(exc_type, exc_value, exc_tb)


def install_hook() -> None:
    """Install the AutoFixer exception hook."""
    global _original_hook
    if _original_hook is None:
        _original_hook = sys.excepthook
    sys.excepthook = _autofixer_excepthook


def uninstall_hook() -> None:
    """Restore the original exception hook."""
    global _original_hook
    if _original_hook is not None:
        sys.excepthook = _original_hook
        _original_hook = None


def get_last_snapshot() -> Optional[ContextSnapshot]:
    """Return the most recently captured ContextSnapshot, or None."""
    return _last_snapshot
