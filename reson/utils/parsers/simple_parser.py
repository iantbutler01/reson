"""
Fallback implementation of OutputParser.

This parser provides a simple JSON-based parsing implementation
for when neither GASP nor BAML are available.
"""

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin

from .base import OutputParser, ParserResult

T = TypeVar('T')


class SimpleParser(OutputParser[T]):
    """
    A simple fallback parser implementation.
    
    This implementation uses basic regex patterns to extract and parse JSON 
    from LLM outputs, falling back to Python's built-in json module.
    """
    
    def parse(self, output: str, output_type: Type[T]) -> ParserResult[T]:
        """
        Parse complete output using simple JSON parsing.
        
        Args:
            output: The complete LLM output
            output_type: The type to parse into
            
        Returns:
            The parsed result
        """
        try:
            # Extract JSON from the output
            json_data = self._extract_json(output)
            
            # Convert to the target type
            value = self._convert_to_type(json_data, output_type)
            
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
        Create a simple streaming parser.
        
        Args:
            output_type: The type to parse into
            
        Returns:
            A dict with parser state
        """
        return {
            "buffer": "",
            "latest_value": None,
            "type": output_type
        }
    
    def feed_chunk(self, parser: Dict[str, Any], chunk: str) -> ParserResult[T]:
        """
        Feed a chunk to the simple parser.
        
        Args:
            parser: The parser state object
            chunk: A chunk of LLM output
            
        Returns:
            The current parsed result
        """
        try:
            # Append to the buffer
            parser["buffer"] += chunk
            
            # Try to parse the current buffer
            try:
                json_data = self._extract_json(parser["buffer"])
                if json_data:
                    value = self._convert_to_type(json_data, parser["type"])
                    parser["latest_value"] = value
                    
                    return ParserResult(
                        value=value,
                        is_partial=True,
                        raw_output=parser["buffer"]
                    )
            except Exception:
                pass
            
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
            parser: The parser state object
            
        Returns:
            The final validated result
        """
        try:
            # Final parse attempt
            json_data = self._extract_json(parser["buffer"])
            if json_data:
                value = self._convert_to_type(json_data, parser["type"])
                return ParserResult(
                    value=value,
                    is_partial=False,
                    raw_output=parser["buffer"]
                )
            
            # If we couldn't extract JSON but have a latest value, use that
            if parser["latest_value"] is not None:
                return ParserResult(
                    value=parser["latest_value"],
                    is_partial=False,
                    raw_output=parser["buffer"]
                )
            
            raise ValueError("Could not extract valid JSON from the output")
        except Exception as e:
            return ParserResult(
                error=e,
                is_partial=False,
                raw_output=parser["buffer"]
            )
    
    def enhance_prompt(self, prompt: str, output_type: Type[T]) -> str:
        """
        Enhance a prompt with simple JSON formatting instructions.
        
        Args:
            prompt: The original prompt
            output_type: The expected output type
            
        Returns:
            The enhanced prompt
        """
        # Generate a schema example
        schema = self._generate_schema_example(output_type)
        schema_str = json.dumps(schema, indent=2)
        
        # Add formatting instructions
        instructions = f"""
{prompt}

Return your response as a JSON object with the following structure:
```json
{schema_str}
```

Make sure your output is valid JSON.
"""
        return instructions.strip()
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text using regex patterns.
        
        Args:
            text: The text to extract JSON from
            
        Returns:
            The extracted JSON as a Python object
        """
        # Try to find JSON inside code blocks
        json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_matches:
            for match in json_matches:
                try:
                    return json.loads(match)
                except Exception:
                    continue
        
        # Try with curly braces
        json_matches = re.findall(r'(\{[\s\S]*\})', text)
        if json_matches:
            for match in json_matches:
                try:
                    return json.loads(match)
                except Exception:
                    continue
        
        # Try to get any structured data using Python's ast.literal_eval
        try:
            import ast
            return ast.literal_eval(text)
        except Exception:
            pass
        
        # Last resort: try the entire text
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # Could not extract JSON
        raise ValueError("Could not extract valid JSON from the output")
    
    def _convert_to_type(self, data: Any, output_type: Type[T]) -> T:
        """
        Convert parsed data to the target type.
        
        Args:
            data: The parsed data
            output_type: The target type
            
        Returns:
            The converted data
        """
        # Handle primitive types
        if output_type in (str, int, float, bool):
            return output_type(data)
        
        # Handle lists
        origin = get_origin(output_type)
        if origin is list:
            args = get_args(output_type)
            if args and args[0] != Any:
                return [self._convert_to_type(item, args[0]) for item in data]
            return data
        
        # Handle dictionaries
        if origin is dict:
            return data
        
        # Handle Pydantic models
        if hasattr(output_type, "model_validate"):  # Pydantic v2
            return output_type.model_validate(data)
        elif hasattr(output_type, "parse_obj"):  # Pydantic v1
            return output_type.parse_obj(data)
        
        # Handle dataclasses
        if hasattr(output_type, "__dataclass_fields__"):
            import dataclasses
            field_types = {f.name: f.type for f in dataclasses.fields(output_type)}
            typed_data = {}
            for key, value in data.items():
                if key in field_types:
                    typed_data[key] = self._convert_to_type(value, field_types[key])
                else:
                    typed_data[key] = value
            return output_type(**typed_data)
        
        # Handle classes with __init__
        if hasattr(output_type, "__init__"):
            return output_type(**data)
        
        # Default: just return the data
        return data
    
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
