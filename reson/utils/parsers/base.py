"""
Base types and interfaces for parsers.

This module defines the abstract interface for output parsers
and common utility types.
"""

import abc
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, get_origin, get_args
import inspect

T = TypeVar('T')


class ParserResult(Generic[T]):
    """
    Result of parsing an LLM output.
    
    This class holds both the parsed value and any metadata
    about the parsing process.
    """
    
    def __init__(
        self,
        value: Optional[T] = None,
        is_partial: bool = False,
        raw_output: Optional[str] = None,
        error: Optional[Exception] = None
    ):
        """
        Initialize a parser result.
        
        Args:
            value: The parsed value, if successful
            is_partial: Whether this is a partial result (for streaming)
            raw_output: The raw output string that was parsed
            error: Any error that occurred during parsing
        """
        self.value = value
        self.is_partial = is_partial
        self.raw_output = raw_output
        self.error = error
        
    @property
    def success(self) -> bool:
        """Whether parsing was successful."""
        return self.error is None and self.value is not None
    
    def __str__(self) -> str:
        if not self.success:
            return f"ParserResult(error={self.error})"
        partial = " (partial)" if self.is_partial else ""
        return f"ParserResult{partial}(value={self.value})"


class OutputParser(Generic[T], abc.ABC):
    """
    Abstract interface for LLM output parsers.
    
    Parsers are responsible for:
    1. Converting LLM outputs to typed Python objects
    2. Supporting both streaming and non-streaming modes
    3. Enhancing prompts with type information (optional)
    """
    
    @abc.abstractmethod
    def parse(self, output: str, output_type: Type[T]) -> ParserResult[T]:
        """
        Parse a complete LLM output into a typed result.
        
        Args:
            output: The complete text output from an LLM
            output_type: The target type to parse into
            
        Returns:
            A ParserResult containing the parsed value or error
        """
        pass
    
    @abc.abstractmethod
    def create_stream_parser(self, output_type: Type[T]) -> Any:
        """
        Create a parser instance for handling streaming output.
        
        Args:
            output_type: The target type to parse into
            
        Returns:
            A parser object that can be fed chunks with feed_chunk
        """
        pass
    
    @abc.abstractmethod
    def feed_chunk(self, parser: Any, chunk: str) -> ParserResult[T]:
        """
        Feed a chunk to a streaming parser and get the current result.
        
        Args:
            parser: A parser instance from create_stream_parser
            chunk: A chunk of text from the LLM stream
            
        Returns:
            A ParserResult with the current parsed value (may be partial)
        """
        pass
    
    @abc.abstractmethod
    def enhance_prompt(self, prompt: str, output_type: Type[T]) -> str:
        """
        Enhance a prompt with type information.
        
        This is used to add schema information, type hints, or
        output format descriptions to the prompt.
        
        Args:
            prompt: The original prompt
            output_type: The expected output type
            
        Returns:
            The enhanced prompt
        """
        pass
    
    def validate_final(self, parser: Any) -> ParserResult[T]:
        """
        Validate and finalize a streaming parser result.
        
        Args:
            parser: A parser instance from create_stream_parser
            
        Returns:
            The final parsed result
        """
        # Default implementation - may be overridden by subclasses
        return ParserResult(
            value=getattr(parser, "value", None),
            is_partial=False,
            raw_output=getattr(parser, "raw_output", None),
            error=getattr(parser, "error", None)
        )
    
    # Type information handling is delegated to the specific parser implementations
    # since GASP and BAML both have built-in type handling capabilities


def get_default_parser() -> OutputParser:
    """
    Get the default OutputParser implementation.
    
    The default is determined by availability:
    1. GASPParser if GASP is installed (should always be available)
    2. BAML parser if BAML is available
    
    Returns:
        An OutputParser instance
    """
    # GASP should always be available in this codebase
    from .gasp_parser import GASPParser
    return GASPParser()
