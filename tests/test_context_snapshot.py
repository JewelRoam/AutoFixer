"""Tests for autofixer.context_snapshot — ContextSnapshot dataclass.

Verifies serialization/deserialization roundtrip and text format output.
"""

import json
import pytest
from autofixer.context_snapshot import ContextSnapshot


@pytest.fixture
def sample_snapshot():
    return ContextSnapshot(
        exception_type="IndexError",
        traceback="Traceback (most recent call last):\n  File \"main.py\", line 42\nIndexError: list index out of range",
        target_file="/home/user/project/main.py",
        crash_line_num=42,
        source_code="def process(items):\n    return items[10]  # BUG: may overflow\n",
        local_vars='{"items": [1, 2, 3], "self": "<MyClass>"}',
    )


class TestContextSnapshotCreation:
    def test_fields_stored(self, sample_snapshot):
        assert sample_snapshot.exception_type == "IndexError"
        assert sample_snapshot.crash_line_num == 42
        assert sample_snapshot.target_file == "/home/user/project/main.py"

    def test_source_code_preserved(self, sample_snapshot):
        assert "items[10]" in sample_snapshot.source_code

    def test_local_vars_is_parseable_json(self, sample_snapshot):
        parsed = json.loads(sample_snapshot.local_vars)
        assert parsed["items"] == [1, 2, 3]


class TestContextSnapshotSerialization:
    def test_json_roundtrip(self, sample_snapshot):
        json_str = sample_snapshot.to_json()
        restored = ContextSnapshot.from_json(json_str)
        assert restored == sample_snapshot

    def test_json_contains_all_fields(self, sample_snapshot):
        json_str = sample_snapshot.to_json()
        data = json.loads(json_str)
        assert set(data.keys()) == {
            "exception_type", "traceback", "target_file",
            "crash_line_num", "source_code", "local_vars",
        }

    def test_to_text_contains_exception(self, sample_snapshot):
        text = sample_snapshot.to_text()
        assert "IndexError" in text

    def test_to_text_contains_source_code(self, sample_snapshot):
        text = sample_snapshot.to_text()
        assert "items[10]" in text

    def test_to_text_contains_local_vars(self, sample_snapshot):
        text = sample_snapshot.to_text()
        assert '"items"' in text

    def test_to_text_contains_file_location(self, sample_snapshot):
        text = sample_snapshot.to_text()
        assert "/home/user/project/main.py:42" in text
