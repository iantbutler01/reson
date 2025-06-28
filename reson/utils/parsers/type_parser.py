"""
Type-based implementation of OutputParser.

This parser uses a type-aware approach to handle structured LLM outputs
with automatic schema detection and validation.
"""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar, Union, get_args, get_origin

from gasp import Parser
from reson.types import Deserializable
from gasp.jinja_helpers import create_type_environment
import jinja2

from .base import OutputParser, ParserResult

T = TypeVar('T')

# Jinja2 environment with type filters
_jinja_env = create_type_environment()
_jinja_env.filters["json"] = lambda obj: json.dumps(obj, indent=2)


class TypeParser(OutputParser[T]):
    """
    Parser implementation using type annotations.
    
    This implementation uses tag-based parsing with type annotations
    to handle LLM outputs.
    """
    
    def parse(self, output: str, output_type: Type[T]) -> ParserResult[T]:
        """
        Parse complete output with type information.
        
        Args:
            output: The complete LLM output
            output_type: The type to parse into
            
        Returns:
            The parsed result
        """
        try:
            # Create a parser for the output type
            parser = self._create_parser(output_type)
            
            # Feed the entire output and validate
            parser.feed(output)
            value = parser.validate()
            
            return ParserResult(
                value=value,
                is_partial=False,
                raw_output=output
            )
        except Exception as e:
            print("WE HIT AN ERROR AND DIED: {e}")
            return ParserResult(
                error=e,
                raw_output=output
            )
    
    def create_stream_parser(self, output_type: Type[T]) -> Any:
        """
        Create a Parser for streaming.
        
        Args:
            output_type: The type to parse into
            
        Returns:
            A tuple of (parser, output_type) for use with feed_chunk
        """
        parser = self._create_parser(output_type)
        # Return both the parser and the output_type so feed_chunk can use it
        return (parser, output_type)
    
    # Class variable to store buffers for parsers
    _parser_buffers = {}
    
    def feed_chunk(self, parser, chunk: str) -> ParserResult[T]:
        """
        Feed a chunk to the Parser.
        
        Args:
            stream_parser: A tuple of (parser, output_type) from create_stream_parser
            chunk: A chunk of LLM output
            
        Returns:
            The current parsed result with properly typed value
        """
        # Unpack the parser and output_type
        parser, output_type = parser
        
        # Use parser's id as a key in our buffer dictionary
        parser_id = id(parser)
        
        # Initialize buffers if they don't exist
        if parser_id not in self._parser_buffers:
            self._parser_buffers[parser_id] = {
                "chunk_buffer": "",
                "debug_buffer": ""
            }
        
        # Update debug buffer (for diagnostic purposes)
        self._parser_buffers[parser_id]["debug_buffer"] += chunk
        
        try:
            # Accumulate the chunk in our buffer
            self._parser_buffers[parser_id]["chunk_buffer"] += chunk
            current_buffer = self._parser_buffers[parser_id]["chunk_buffer"]

            def truncate_to_valid_utf8(data: str) -> str:
                encoded = data.encode("utf-8", errors="ignore")
                decoded = encoded.decode("utf-8", errors="ignore")
                return decoded

            # Defensive approach to handle UTF-8 boundary issues
            try:
                # Feed the accumulated buffer to the parser
                safe_buffer = truncate_to_valid_utf8(current_buffer)
                raw_result = parser.feed(safe_buffer)
                # If successful, clear chunk buffer
                self._parser_buffers[parser_id]["chunk_buffer"] = ""
            
                
                # Convert to typed object if needed
                result = self._convert_to_typed_result(raw_result, output_type)
            except Exception as utf8_err:
                # In case of UTF-8 boundary errors, keep collecting chunks
                # We'll try again when more data arrives
                print(f"Parser chunk error (likely UTF-8 boundary): {utf8_err}")
                return ParserResult(
                    value=None,
                    is_partial=True,
                    raw_output=self._parser_buffers[parser_id]["debug_buffer"]
                )

            
            # Get raw output from debug buffer
            raw_output = self._parser_buffers[parser_id]["debug_buffer"]
            
            # For streaming, we don't validate yet
            return ParserResult(
                value=result,
                is_partial=True,
                raw_output=raw_output
            )
        except Exception as e:
            print(f"Parser general error: {e}")
            return ParserResult(
                error=e,
                is_partial=True,
                raw_output=self._parser_buffers[parser_id]["debug_buffer"]
            )
    
    def validate_final(self, parser) -> ParserResult[T]:
        """
        Validate a streaming parser at the end.
        
        Args:
            stream_parser: A tuple of (parser, output_type) from create_stream_parser
            
        Returns:
            The final validated result
        """
        # Unpack the parser and output_type
        parser, output_type = parser
        
        try:
            # Validate the final result
            raw_value = parser.validate()
            
            # Ensure we have the proper typed object
            value = self._convert_to_typed_result(raw_value, output_type)
            
            return ParserResult(
                value=value,
                is_partial=False,
                raw_output=getattr(parser, "_buffer", "")
            )
        except Exception as e:
            print(f"Validation error: {e}")
            return ParserResult(
                error=e,
                is_partial=False,
                raw_output=getattr(parser, "_buffer", "")
            )
    
    def enhance_prompt(self, prompt: str, output_type: Type[T], call_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance a prompt with type information and Jinja2 rendering.
        
        Args:
            prompt: The original prompt
            output_type: The expected output type
            call_context: Optional dictionary of arguments for Jinja2 rendering
            
        Returns:
            The enhanced and rendered prompt
        """
        # If output_type is None, and no call_context for Jinja, return original prompt
        if output_type is None and not call_context:
            return prompt

        processed_prompt = prompt
        
        # Stage 1: Interpolate {{return_type}} using GASP
        if output_type is not None:
            try:
                from gasp.template_helpers import interpolate_prompt
                # Ensure {return_type} becomes {{return_type}} for gasp's interpolate_prompt
                # This step specifically handles the {{return_type}} placeholder.
                print(output_type)
                processed_prompt = interpolate_prompt(processed_prompt, output_type, format_tag="return_type")
            except Exception as e:
                print(f"Error enhancing prompt with type information (interpolate_prompt): {e}")
                # Continue with the current prompt if type interpolation fails
        
        # Stage 2: Render with Jinja2 using call_context if provided
        if call_context:
            try:
                # Ensure the _jinja_env is initialized (it should be by class instantiation)
                if not hasattr(self, '_jinja_env') or self._jinja_env is None:
                    # This is a fallback, should ideally not be needed if constructor sets it.
                    self._jinja_env = create_type_environment()
                    self._jinja_env.filters["json"] = lambda obj: json.dumps(obj, indent=2)
                
                template = self._jinja_env.from_string(processed_prompt)
                processed_prompt = template.render(**call_context)
            except Exception as e:
                print(f"Error rendering prompt with Jinja2: {e}")
                # Return the prompt as it was before Jinja rendering if Jinja fails

        return processed_prompt
    
    def _create_parser(self, output_type: Type[T]) -> Parser:
        """
        Create a Parser for a specific type.
        
        Args:
            output_type: The type to parse into
            
        Returns:
            A Parser instance
        """
        # Get the origin type for generics (List, Dict, etc)
        origin_type = get_origin(output_type)
        
        # Handle generic types
        if origin_type is not None:
            # For container types like List[T], Dict[K, V], etc.
            return Parser(output_type)
        
        # Check if the type is a Pydantic model (non-generic only)
        if hasattr(output_type, "model_validate") or hasattr(output_type, "parse_obj"):
            return Parser.from_pydantic(output_type)
        
        # For Deserializable classes (non-generic only)
        # Only use issubclass for actual classes, not generic types
        try:
            if hasattr(output_type, "__mro__") and Deserializable in output_type.__mro__:
                return Parser(output_type)
        except TypeError:
            # In case output_type isn't a class
            pass
        
        # For Python standard types (non-generic)
        return Parser(output_type)
    
    def _convert_to_typed_result(self, result: Any, output_type: Type[T]) -> Any:
        """
        Convert a result to the proper typed object if needed, handling nested objects.
        
        For Pydantic models, the parser may return a dict during streaming.
        We need to convert this to a properly typed object, including all nested models.
        
        Args:
            result: The result from parser.feed()
            output_type: The target type
            
        Returns:
            A properly typed object with all nested objects also properly typed
        """
        # Handle None result
        if result is None:
            return None
            
        # Get the origin type for generics (List, Dict, etc)
        origin_type = get_origin(output_type)
        type_args = get_args(output_type)
        
        # Non-generic Pydantic models
        if hasattr(output_type, "model_construct") and origin_type is None:
            # Direct instance check for non-generic types (safe)
            if type(result) == output_type:
                return result
                
            # If result is a dict, convert it to the proper Pydantic model
            if isinstance(result, dict):
                try:
                    # Get the model's field types
                    model_fields = getattr(output_type, "model_fields", None)
                    field_types = {}
                    
                    # For Pydantic v2
                    if model_fields:
                        for field_name, field_info in model_fields.items():
                            field_types[field_name] = field_info.annotation
                    # Fallback to __annotations__ if model_fields not available
                    elif hasattr(output_type, "__annotations__"):
                        field_types = output_type.__annotations__
                    
                    # Convert nested objects recursively
                    converted_dict = {}
                    for key, value in result.items():
                        # Skip fields not in the model
                        if key not in field_types:
                            converted_dict[key] = value
                            continue
                        
                        # Get the expected type for this field
                        field_type = field_types.get(key)
                        if field_type is not None:
                            # Recursively convert the field value
                            converted_dict[key] = self._convert_to_typed_result(value, field_type)
                        else:
                            converted_dict[key] = value
                    
                    # Convert to proper type using model_construct without validation

                    return output_type.model_construct(**converted_dict) # type: ignore
                except Exception as e:
                    print(f"Error converting dict to {output_type.__name__}: {e}")
                    return result
        
        # Handle List[T] case - check against the runtime type 'list', not the generic List[T]
        if origin_type == list and isinstance(result, list) and type_args:
            item_type = type_args[0]
            # Try to convert each item recursively
            try:
                return [self._convert_to_typed_result(item, item_type) for item in result]
            except Exception as e:
                print(f"Error converting list items: {e}")
                return result
        
        # Handle Dict[K, V] case - check against the runtime type 'dict', not the generic Dict[K,V]
        if origin_type == dict and isinstance(result, dict) and len(type_args) == 2:
            value_type = type_args[1]
            # Only convert values, not keys
            try:
                return {k: self._convert_to_typed_result(v, value_type) for k, v in result.items()}
            except Exception as e:
                print(f"Error converting dict values: {e}")
                return result
        
        # For primitive types, still use isinstance
        if output_type in (str, int, float, bool) and isinstance(result, output_type):
            return result
                
        # For other cases, return as is
        return result

    def _generate_schema_example(self, output_type: Type[T]) -> Any:
        """
        Generate a schema example for the output type.
        
        Args:
            output_type: The type to generate a schema for
            
        Returns:
            A dict or other structure representing the schema
        """
        # Handle primitive types
        if output_type in (str, int, float, bool):
            return f"{output_type.__name__.lower()} value"
        
        # Handle lists and other generic containers
        origin = get_origin(output_type)
        if origin is not None:
            args = get_args(output_type)
            
            # Handle List[X]
            if origin == list:
                if args and args[0] != Any:
                    return [self._generate_schema_example(args[0])]
                return ["item1", "item2"]
            
            # Handle Dict[K, V]
            if origin == dict:
                if len(args) == 2 and args[0] != Any and args[1] != Any:
                    key_example = "key" if args[0] == str else "1"
                    return {key_example: self._generate_schema_example(args[1])}
                return {"key1": "value1", "key2": "value2"}
            
            # Handle Optional[X]
            if origin == Union and type(None) in args:
                non_none_args = [arg for arg in args if arg != type(None)]
                if non_none_args:
                    return self._generate_schema_example(non_none_args[0])
        
        # Handle classes with annotations (like Pydantic models)
        if hasattr(output_type, "__annotations__"):
            result = {}
            for field_name, field_type in output_type.__annotations__.items():
                result[field_name] = self._generate_schema_example(field_type)
            return result
        
        # Default fallback
        return "..."
