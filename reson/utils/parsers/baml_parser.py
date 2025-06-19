"""
BAML-based implementation of OutputParser.

This parser uses BAML's modular API for handling LLM outputs.
It provides integration with BAML's client and parsing capabilities.
"""

import json
from typing import Any, Dict, Optional, Type, TypeVar, Union

from .base import OutputParser, ParserResult

T = TypeVar('T')


def _extract_baml_text(request) -> str:
    """
    Given a BAML stream_request object, return the plain-text content of the
    first system message so it can be fed into our ChatMessage structure.

    The expected structure is like:
    {
      "model": "...",
      "messages": [
        {
          "role": "system",
          "content": [
            {"type": "text", "text": "... actual prompt ..."}
          ]
        },
        ...
      ],
      ...
    }
    """
    body = request.body.json()  # as provided by b.stream_request.*

    # Body may already be a dict or JSON string
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            # Couldn't JSON-decode – assume it is already the text
            return body

    if isinstance(body, dict):
        msgs = body.get("messages") or []
        if msgs:
            first_msg = msgs[0]  # only the system message matters here
            content = first_msg.get("content") or []
            if content and isinstance(content, list):
                part = content[0]
                if isinstance(part, dict):
                    text = part.get("text")
                    if text is not None:
                        # Remove any leading/trailing triple quotes artefacts
                        return text.lstrip('"').lstrip("\n").rstrip('"')
    # Fallback – stringify whatever is left
    return str(body)


class BAMLParser(OutputParser[T]):
    """
    Parser implementation using BAML.
    
    This implementation integrates with BAML's modular API to handle
    LLM outputs with type safety.
    """
    
    def __init__(self):
        """Initialize the BAML parser."""
        try:
            import baml_client
            self.baml = baml_client
        except ImportError:
            raise ImportError(
                "BAML client is not installed. "
                "Please install it with `pip install baml-client`."
            )
    
    def parse(self, output: str, output_type: Type[T]) -> ParserResult[T]:
        """
        Parse complete output with BAML.
        
        Args:
            output: The complete LLM output
            output_type: The type to parse into
            
        Returns:
            The parsed result
        """
        try:
            # Get the function name from the type
            type_name = getattr(output_type, "__name__", str(output_type))
            
            # Use BAML's parsing capabilities
            value = self.baml.parse.__getattr__(type_name)(
                {"choices": [{"message": {"content": output}}]}
            )
            
            return ParserResult(
                value=value,
                is_partial=False,
                raw_output=output
            )
        except Exception as e:
            return ParserResult(
                error=e,
                raw_output=output
            )
    
    def create_stream_parser(self, output_type: Type[T]) -> Any:
        """
        Create a BAML streaming parser.
        
        Args:
            output_type: The type to parse into
            
        Returns:
            A structure containing parser state
        """
        # Create a simple state object
        return {
            "type_name": getattr(output_type, "__name__", str(output_type)),
            "buffer": "",
            "latest_value": None,
            "type": output_type
        }
    
    def feed_chunk(self, parser: Dict[str, Any], chunk: str) -> ParserResult[T]:
        """
        Feed a chunk to the BAML streaming parser.
        
        Args:
            parser: The BAML parser state object
            chunk: A chunk of LLM output
            
        Returns:
            The current parsed result
        """
        try:
            # Append to the buffer
            parser["buffer"] += chunk
            
            # Try to parse the current buffer
            try:
                value = self.baml.parse.__getattr__(parser["type_name"])(
                    {"choices": [{"message": {"content": parser["buffer"]}}]}
                )
                parser["latest_value"] = value
                
                return ParserResult(
                    value=value,
                    is_partial=True,
                    raw_output=parser["buffer"]
                )
            except Exception:
                # Return the latest successful value
                return ParserResult(
                    value=parser["latest_value"],
                    is_partial=True,
                    raw_output=parser["buffer"]
                )
        except Exception as e:
            return ParserResult(
                error=e,
                is_partial=True,
                raw_output=parser["buffer"]
            )
    
    def validate_final(self, parser: Dict[str, Any]) -> ParserResult[T]:
        """
        Validate a streaming parser at the end.
        
        Args:
            parser: The BAML parser state object
            
        Returns:
            The final validated result
        """
        try:
            # Final parse attempt
            value = self.baml.parse.__getattr__(parser["type_name"])(
                {"choices": [{"message": {"content": parser["buffer"]}}]}
            )
            
            return ParserResult(
                value=value,
                is_partial=False,
                raw_output=parser["buffer"]
            )
        except Exception as e:
            # If final parse fails, return the latest successful value
            if parser["latest_value"] is not None:
                return ParserResult(
                    value=parser["latest_value"],
                    is_partial=False,
                    raw_output=parser["buffer"]
                )
            
            return ParserResult(
                error=e,
                is_partial=False,
                raw_output=parser["buffer"]
            )
    
    def enhance_prompt(self, prompt: str, output_type: Type[T], call_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the prompt template from BAML.
        
        Args:
            prompt: The original prompt
            output_type: The expected output type
            call_context: Optional dictionary of arguments (ignored by BAMLParser)
            
        Returns:
            The enhanced prompt
        """
        # For BAML, we don't modify the prompt directly
        # because BAML handles prompt templating itself. call_context is ignored.
        return prompt
    
    def get_baml_request(self, prompt: str, output_type: Type[T]) -> Any:
        """
        Get a BAML request object for the prompt and type.
        
        Args:
            prompt: The prompt to use
            output_type: The type to parse into
            
        Returns:
            A BAML request object
        """
        type_name = getattr(output_type, "__name__", str(output_type))
        
        # Create a request for the specific type
        # The "Extract{Type}" naming is conventional in BAML
        return getattr(self.baml, f"Extract{type_name}")(prompt)
    
    def extract_prompt_from_baml_request(self, request: Any) -> str:
        """
        Extract the prompt text from a BAML request.
        
        Args:
            request: A BAML request object
            
        Returns:
            The text of the prompt
        """
        return _extract_baml_text(request)
