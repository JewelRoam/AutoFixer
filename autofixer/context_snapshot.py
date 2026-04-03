"""ContextSnapshot: the structured input for AutoFixer forward pass.

Corresponds to the Viba schema:
    ContextSnapshot := dict[
        'exception_type' = str,
        'traceback' = str,
        'target_file' = str,
        'crash_line_num' = int,
        'source_code' = str,
        'local_vars' = str
    ]
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class ContextSnapshot:
    exception_type: str
    traceback: str
    target_file: str
    crash_line_num: int
    source_code: str
    local_vars: str

    def to_text(self) -> str:
        """Serialize to a human-readable text block for symbolic tensor storage."""
        lines = [
            f"## Exception: {self.exception_type}",
            f"## File: {self.target_file}:{self.crash_line_num}",
            "",
            "### Traceback",
            self.traceback,
            "",
            "### Source Code",
            self.source_code,
            "",
            "### Local Variables",
            self.local_vars,
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ContextSnapshot":
        """Deserialize from JSON string."""
        return cls(**json.loads(text))
