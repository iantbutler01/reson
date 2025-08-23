"""Base class for generating provider-specific tool schemas from Deserializable types."""

from abc import ABC, abstractmethod
from typing import (
    Dict,
    Any,
    Callable,
    Type,
    List,
    get_type_hints,
    get_origin,
    get_args,
    Union,
)
import inspect


class SchemaGenerator(ABC):
    """Base class for generating provider-specific tool schemas from Deserializable types."""

    @abstractmethod
    def generate_tool_schemas(self, tools: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Generate list of tool schemas in provider-specific format."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name this generator supports."""
        pass

    def _extract_function_info(self, func: Callable) -> Dict[str, Any]:
        """Extract function signature, parameters, and docstring info."""
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        parameters = {}
        required = []

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, str)
            param_info = self._python_type_to_schema(param_type)

            # Add description from parameter annotation if available
            if hasattr(param, "annotation") and hasattr(param.annotation, "__doc__"):
                param_info["description"] = param.annotation.__doc__

            parameters[param_name] = param_info

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": func.__name__,
            "description": func.__doc__ or f"Execute {func.__name__}",
            "parameters": parameters,
            "required": required,
        }

    def _python_type_to_schema(self, python_type: Type) -> Dict[str, Any]:
        """Convert Python type to JSON schema format."""
        origin = get_origin(python_type)
        args = get_args(python_type)

        # Handle Union types (including Optional)
        if origin is Union:
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                # Optional[T] case
                return self._python_type_to_schema(non_none_types[0])
            else:
                # Union[A, B, ...] case - use first type as primary
                return self._python_type_to_schema(non_none_types[0])

        # Handle List/Array types
        if origin in (list, List):
            item_type = args[0] if args else str
            return {"type": "array", "items": self._python_type_to_schema(item_type)}

        # Handle Dict/Object types
        if origin in (dict, Dict):
            return {"type": "object"}

        # Handle Deserializable types
        if hasattr(python_type, "__mro__") and any(
            "Deserializable" in cls.__name__ for cls in python_type.__mro__
        ):
            return self._deserializable_to_schema(python_type)

        # Handle primitive types
        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
        }

        return type_mapping.get(python_type, {"type": "string"})

    def _deserializable_to_schema(self, deserializable_class: Type) -> Dict[str, Any]:
        """Convert Deserializable class to JSON schema object."""
        type_hints = get_type_hints(deserializable_class)
        properties = {}
        required = []

        # Get field information from class annotations
        for field_name, field_type in type_hints.items():
            if field_name.startswith("_"):
                continue

            properties[field_name] = self._python_type_to_schema(field_type)

            # Check if field has default value
            if not hasattr(deserializable_class, field_name):
                required.append(field_name)

        return {"type": "object", "properties": properties, "required": required}
