"""
Parser abstraction for structured output parsing.

This module provides parsers for handling LLM output in both streaming
and non-streaming contexts, with support for type inference.
"""

from .base import OutputParser, ParserResult, get_default_parser
from .gasp_parser import GASPParser

# Conditionally import BAML parser if BAML is installed
try:
    from .baml_parser import BAMLParser
    __all__ = ["OutputParser", "ParserResult", "get_default_parser", "GASPParser", "BAMLParser"]
except ImportError:
    __all__ = ["OutputParser", "ParserResult", "get_default_parser", "GASPParser"]
