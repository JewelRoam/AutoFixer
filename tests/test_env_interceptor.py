"""Tests for autofixer.env_interceptor — frame_inspector and sys_excepthook.

These tests verify that the interceptor can:
1. Extract local variables from a stack frame
2. Extract source code around a crash line
3. Build a complete ContextSnapshot from an exception
"""

import sys
import json
import pytest
from autofixer.context_snapshot import ContextSnapshot


class TestFrameInspector:
    """Tests for frame_inspector.extract_frame_info()."""

    def test_extract_local_vars_from_frame(self):
        from autofixer.env_interceptor.frame_inspector import extract_local_vars
        # Create a real frame by triggering an exception
        x = 42
        name = "test"
        local_dict = {"x": x, "name": name}
        result = extract_local_vars(local_dict)
        parsed = json.loads(result)
        assert parsed["x"] == 42
        assert parsed["name"] == "test"

    def test_extract_local_vars_handles_unserializable(self):
        from autofixer.env_interceptor.frame_inspector import extract_local_vars
        # Objects that can't be JSON-serialized should be repr()'d
        local_dict = {"obj": object(), "num": 1}
        result = extract_local_vars(local_dict)
        parsed = json.loads(result)
        assert "object" in parsed["obj"]  # repr of object()
        assert parsed["num"] == 1

    def test_extract_source_code(self, tmp_path):
        from autofixer.env_interceptor.frame_inspector import extract_source_code
        # Create a temp source file
        src = tmp_path / "sample.py"
        lines = [f"line_{i}\n" for i in range(50)]
        src.write_text("".join(lines))

        # Extract around line 25 with window=5
        code = extract_source_code(str(src), line_num=25, window=5)
        assert "line_24" in code
        assert "line_25" in code
        assert "line_20" in code
        assert "line_29" in code

    def test_extract_source_code_near_file_start(self, tmp_path):
        from autofixer.env_interceptor.frame_inspector import extract_source_code
        src = tmp_path / "short.py"
        src.write_text("a = 1\nb = 2\nc = 3\n")
        code = extract_source_code(str(src), line_num=1, window=5)
        assert "a = 1" in code

    def test_extract_source_code_file_not_found(self):
        from autofixer.env_interceptor.frame_inspector import extract_source_code
        code = extract_source_code("/nonexistent/file.py", line_num=1, window=5)
        assert code == ""  # graceful degradation


class TestBuildSnapshotFromException:
    """Tests for frame_inspector.build_snapshot_from_exception()."""

    def test_build_snapshot_returns_context_snapshot(self):
        from autofixer.env_interceptor.frame_inspector import build_snapshot_from_exception
        try:
            items = [1, 2, 3]
            _ = items[10]
        except IndexError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            snapshot = build_snapshot_from_exception(exc_type, exc_value, exc_tb)
            assert isinstance(snapshot, ContextSnapshot)
            assert snapshot.exception_type == "IndexError"
            assert snapshot.crash_line_num > 0
            assert "items" in snapshot.local_vars

    def test_build_snapshot_traceback_not_empty(self):
        from autofixer.env_interceptor.frame_inspector import build_snapshot_from_exception
        try:
            1 / 0
        except ZeroDivisionError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            snapshot = build_snapshot_from_exception(exc_type, exc_value, exc_tb)
            assert len(snapshot.traceback) > 0
            assert "ZeroDivisionError" in snapshot.traceback


class TestSysExcepthook:
    """Tests for sys_excepthook.install_hook() / uninstall_hook()."""

    def test_install_and_uninstall_hook(self):
        from autofixer.env_interceptor.sys_excepthook import install_hook, uninstall_hook
        original = sys.excepthook
        install_hook()
        assert sys.excepthook is not original
        uninstall_hook()
        assert sys.excepthook is original

    def test_hook_captures_snapshot(self):
        from autofixer.env_interceptor.sys_excepthook import install_hook, uninstall_hook, get_last_snapshot
        install_hook()
        try:
            # Simulate an unhandled exception by calling the hook directly
            try:
                raise ValueError("test error")
            except ValueError:
                exc_type, exc_value, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_value, exc_tb)

            snapshot = get_last_snapshot()
            assert snapshot is not None
            assert snapshot.exception_type == "ValueError"
        finally:
            uninstall_hook()
