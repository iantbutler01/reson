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
from types import UnionType
import warnings


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
        # Prefer an attached tool_type (if any) as the source of truth for schema/description
        tool_type = getattr(func, "__reson_tool_type__", None)
        if tool_type is not None:
            object_schema = self._python_type_to_schema(tool_type)
            if (
                not isinstance(object_schema, dict)
                or object_schema.get("type") != "object"
            ):
                raise ValueError(
                    f"tool_type for {func.__name__} must resolve to an object schema"
                )
            description = (
                getattr(tool_type, "__doc__", None)
                or func.__doc__
                or f"Execute {func.__name__}"
            )
            return {
                "name": func.__name__,
                "description": description,
                "parameters": object_schema.get("properties", {}),
                "required": object_schema.get("required", []),
            }

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
            if (
                hasattr(param, "annotation")
                and hasattr(param.annotation, "__doc__")
                and param.annotation.__doc__ is not None
            ):
                param_info["description"] = param.annotation.__doc__
            elif (
                "description" not in param_info or param_info.get("description") is None
            ):
                # Ensure description is never null for JSON Schema 2020-12 compliance
                param_info["description"] = (
                    f"Parameter {param_name} of type {param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)}"
                )

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

        # Handle Union types (including Optional and PEP 604 unions)
        if origin in (Union, UnionType):
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                # Optional[T] case
                return self._python_type_to_schema(non_none_types[0])
            else:
                # Union[A, B, ...] case - providers generally don't support oneOf;
                # collapse to the first non-None type but warn to aid debugging.
                try:
                    type_names = ", ".join(
                        getattr(t, "__name__", str(t)) for t in non_none_types
                    )
                except Exception:
                    type_names = str(non_none_types)
                warnings.warn(
                    f"[reson.schema] Collapsing Union[{type_names}] to first type "
                    f"{getattr(non_none_types[0], '__name__', str(non_none_types[0]))} for schema generation",
                    UserWarning,
                )
                return self._python_type_to_schema(non_none_types[0])

        # Handle List/Array types
        if origin in (list, List):
            item_type = args[0] if args else str
            return {"type": "array", "items": self._python_type_to_schema(item_type)}

        # Handle Dict/Object types
        if origin in (dict, Dict):
            if len(args) >= 2:
                # Dict[str, T] - add proper additionalProperties
                value_type = args[1]
                return {
                    "type": "object",
                    "additionalProperties": self._python_type_to_schema(value_type),
                }
            return {"type": "object", "additionalProperties": True}

        # Handle Deserializable types first (your core type system)
        if (
            hasattr(python_type, "__mro__")
            and inspect.isclass(python_type)
            and any("Deserializable" in cls.__name__ for cls in python_type.__mro__)
        ):
            return self._deserializable_to_schema(python_type)

        # Handle Pydantic models (check for actual BaseModel inheritance)
        if (
            inspect.isclass(python_type)
            and hasattr(python_type, "__mro__")
            and any("BaseModel" in cls.__name__ for cls in python_type.__mro__)
        ):
            return self._pydantic_to_schema(python_type)

        # Handle Dataclasses (check for dataclass decorator)
        if inspect.isclass(python_type) and hasattr(
            python_type, "__dataclass_fields__"
        ):
            return self._dataclass_to_schema(python_type)

        # Handle regular Python classes (only if not builtin and has custom __init__)
        if (
            inspect.isclass(python_type)
            and hasattr(python_type, "__init__")
            and python_type.__module__ != "builtins"
            and python_type not in (str, int, float, bool, list, dict, tuple, set)
        ):
            return self._regular_class_to_schema(python_type)

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

            field_schema = self._python_type_to_schema(field_type)
            # Ensure description is never null for JSON Schema 2020-12 compliance
            if "description" not in field_schema or field_schema["description"] is None:
                field_schema["description"] = (
                    f"Field {field_name} of type {field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}"
                )

            properties[field_name] = field_schema

            # Check if field has default value
            if not hasattr(deserializable_class, field_name):
                required.append(field_name)

        schema = {"type": "object", "properties": properties, "required": required}
        # Ensure the object itself has a description
        if "description" not in schema:
            schema["description"] = f"Object of type {deserializable_class.__name__}"

        return schema

    def _pydantic_to_schema(self, pydantic_class: Type) -> Dict[str, Any]:
        """Convert Pydantic model to JSON schema object."""
        properties = {}
        required = []

        # Pydantic v2
        if hasattr(pydantic_class, "model_fields"):
            for field_name, field_info in pydantic_class.model_fields.items():
                field_type = field_info.annotation
                field_schema = self._python_type_to_schema(field_type)
                # Ensure description is never null
                if (
                    "description" not in field_schema
                    or field_schema["description"] is None
                ):
                    field_schema["description"] = (
                        f"Field {field_name} of type {field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}"
                    )
                properties[field_name] = field_schema

                if field_info.is_required():
                    required.append(field_name)
        # Pydantic v1
        elif hasattr(pydantic_class, "__fields__"):
            for field_name, field_info in pydantic_class.__fields__.items():
                field_type = field_info.type_
                field_schema = self._python_type_to_schema(field_type)
                # Ensure description is never null
                if (
                    "description" not in field_schema
                    or field_schema["description"] is None
                ):
                    field_schema["description"] = (
                        f"Field {field_name} of type {field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}"
                    )
                properties[field_name] = field_schema

                if field_info.is_required():
                    required.append(field_name)
        else:
            # Fallback to annotations
            type_hints = get_type_hints(pydantic_class)
            for field_name, field_type in type_hints.items():
                if field_name.startswith("_"):
                    continue
                field_schema = self._python_type_to_schema(field_type)
                # Ensure description is never null
                if (
                    "description" not in field_schema
                    or field_schema["description"] is None
                ):
                    field_schema["description"] = (
                        f"Field {field_name} of type {field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}"
                    )
                properties[field_name] = field_schema
                if not hasattr(pydantic_class, field_name):
                    required.append(field_name)

        return {"type": "object", "properties": properties, "required": required}

    def _dataclass_to_schema(self, dataclass_type: Type) -> Dict[str, Any]:
        """Convert dataclass to JSON schema object."""
        properties = {}
        required = []

        # Get field information from dataclass fields
        for field_name, field_info in dataclass_type.__dataclass_fields__.items():
            field_type = field_info.type
            field_schema = self._python_type_to_schema(field_type)
            # Ensure description is never null
            if "description" not in field_schema or field_schema["description"] is None:
                field_schema["description"] = (
                    f"Field {field_name} of type {field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}"
                )
            properties[field_name] = field_schema

            # Check if field has no default value
            import dataclasses

            if (
                field_info.default is dataclasses.MISSING
                and field_info.default_factory is dataclasses.MISSING
            ):
                required.append(field_name)

        return {"type": "object", "properties": properties, "required": required}

    def _regular_class_to_schema(self, class_type: Type) -> Dict[str, Any]:
        """Convert regular Python class to JSON schema object using __init__ signature."""
        properties = {}
        required = []

        # Get the __init__ method signature
        init_method = class_type.__init__
        signature = inspect.signature(init_method)
        type_hints = get_type_hints(init_method)

        for param_name, param in signature.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = type_hints.get(param_name, str)
            field_schema = self._python_type_to_schema(param_type)

            # Ensure description is never null
            if "description" not in field_schema or field_schema["description"] is None:
                field_schema["description"] = (
                    f"Field {param_name} of type {param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)}"
                )

            properties[param_name] = field_schema

            # Check if parameter has no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}
