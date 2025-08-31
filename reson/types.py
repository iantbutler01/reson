"""
Type definitions for reson.

This module provides the base types needed for building structured
outputs with reson.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

from gasp import Deserializable
from reson.services.inference_clients import ToolResult, ChatMessage, ReasoningSegment

__all__ = [
    "Deserializable",
    "ToolResult",
    "ChatMessage",
    "ReasoningSegment",
]
