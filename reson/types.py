"""
Type definitions for reson.

This module provides the base types needed for building structured
outputs with reson.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

from gasp import Deserializable


@dataclass
class NativeToolCall:
    """Represents a native tool call from an LLM provider."""

    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str  # Original JSON string for debugging

    def to_deserializable_object(self, tool_func: Callable) -> Any:
        """Convert arguments to proper Deserializable object."""
        # This will be implemented in the parsing logic
        pass


@dataclass
class NativeToolResult:
    """Represents the result of executing a native tool call."""

    tool_call_id: str
    content: Any
    success: bool = True
    error: Optional[str] = None


# Re-export Deserializable for use in reson projects
__all__ = ["Deserializable", "NativeToolCall", "NativeToolResult"]
