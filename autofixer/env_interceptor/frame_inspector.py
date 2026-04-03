"""Frame inspector: extract crash context from Python exception frames.

Provides utilities to build a ContextSnapshot from a live exception,
including local variable extraction and source code windowing.
"""

import json
import linecache
import os
import traceback as tb_module
from types import TracebackType
from typing import Dict, Optional, Type

from autofixer.context_snapshot import ContextSnapshot


def extract_local_vars(local_dict: Dict) -> str:
    """Serialize a frame's local variables to JSON string.

    Non-serializable objects are converted via repr().
    """
    safe = {}
    for key, value in local_dict.items():
        if key.startswith("__"):
            continue
        try:
            json.dumps(value)
            safe[key] = value
        except (TypeError, ValueError, OverflowError):
            safe[key] = repr(value)
    return json.dumps(safe, ensure_ascii=False, indent=2)


def extract_source_code(filepath: str, line_num: int, window: int = 20) -> str:
    """Extract source code around a given line number.

    Returns lines [line_num - window, line_num + window] with line numbers.
    Returns empty string if file cannot be read.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, IOError):
        return ""

    if not lines:
        return ""

    start = max(0, line_num - 1 - window)
    end = min(len(lines), line_num + window)
    numbered = []
    for i in range(start, end):
        marker = " >>> " if i == line_num - 1 else "     "
        numbered.append(f"{i + 1:4d}{marker}{lines[i].rstrip()}")
    return "\n".join(numbered)


def build_snapshot_from_exception(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> ContextSnapshot:
    """Build a ContextSnapshot from exception info (as from sys.exc_info()).

    Walks the traceback to the innermost frame and extracts:
    - exception type and full traceback text
    - source code around the crash line
    - local variables of the crash frame
    """
    # Format full traceback
    tb_text = "".join(tb_module.format_exception(exc_type, exc_value, exc_tb))

    # Walk to innermost frame
    frame_tb = exc_tb
    while frame_tb and frame_tb.tb_next:
        frame_tb = frame_tb.tb_next

    if frame_tb is None:
        return ContextSnapshot(
            exception_type=exc_type.__name__,
            traceback=tb_text,
            target_file="<unknown>",
            crash_line_num=0,
            source_code="",
            local_vars="{}",
        )

    frame = frame_tb.tb_frame
    filename = frame.f_code.co_filename
    lineno = frame_tb.tb_lineno
    local_vars_str = extract_local_vars(dict(frame.f_locals))
    source = extract_source_code(filename, lineno)

    return ContextSnapshot(
        exception_type=exc_type.__name__,
        traceback=tb_text,
        target_file=filename,
        crash_line_num=lineno,
        source_code=source,
        local_vars=local_vars_str,
    )
