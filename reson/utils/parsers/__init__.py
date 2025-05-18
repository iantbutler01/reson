"""
Parser abstraction for structured output parsing.

This module provides parsers for handling LLM output in both streaming
and non-streaming contexts, with support for type inference.
"""

from .base import OutputParser, ParserResult, get_default_parser
from .type_parser import TypeParser

# Conditionally import BAML parser if BAML is installed
try:
    from .baml_parser import BAMLParser
    __all__ = ["OutputParser", "ParserResult", "get_default_parser", "TypeParser", "BAMLParser"]
except ImportError:
    __all__ = ["OutputParser", "ParserResult", "get_default_parser", "TypeParser"]
