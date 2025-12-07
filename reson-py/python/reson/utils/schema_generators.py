"""Schema generators with Google types conversion."""

from reson.reson import (
    SchemaGenerator as _RustSchemaGenerator,
    supports_native_tools,
)

try:
    from google.genai import types as gtypes
except ImportError:
    gtypes = None


def _convert_to_google_schema(schema_dict):
    """Convert a JSON schema dict to google.genai.types.Schema."""
    if gtypes is None:
        raise ImportError(
            "Google GenAI library not available. Install with: pip install google-genai"
        )

    kwargs = {}

    # Map type string to Google Type enum
    type_str = schema_dict.get("type")
    if type_str == "string":
        kwargs["type"] = gtypes.Type.STRING
    elif type_str == "integer":
        kwargs["type"] = gtypes.Type.INTEGER
    elif type_str == "number":
        kwargs["type"] = gtypes.Type.NUMBER
    elif type_str == "boolean":
        kwargs["type"] = gtypes.Type.BOOLEAN
    elif type_str == "array":
        kwargs["type"] = gtypes.Type.ARRAY
        if "items" in schema_dict:
            kwargs["items"] = _convert_to_google_schema(schema_dict["items"])
    elif type_str == "object":
        kwargs["type"] = gtypes.Type.OBJECT
        if "properties" in schema_dict:
            kwargs["properties"] = {
                k: _convert_to_google_schema(v)
                for k, v in schema_dict["properties"].items()
            }
        if "required" in schema_dict:
            kwargs["required"] = schema_dict["required"]
        if "additionalProperties" in schema_dict:
            ap = schema_dict["additionalProperties"]
            if isinstance(ap, bool):
                kwargs["additional_properties"] = ap
            elif isinstance(ap, dict):
                kwargs["additional_properties"] = _convert_to_google_schema(ap)

    if "description" in schema_dict:
        kwargs["description"] = schema_dict["description"]

    return gtypes.Schema(**kwargs)


class SchemaGenerator:
    """Wrapper around Rust SchemaGenerator that converts Google schemas."""

    def __init__(self, provider: str):
        self._rust_generator = _RustSchemaGenerator(provider)
        self._provider = provider

    def generate_tool_schemas(self, tools):
        """Generate tool schemas, converting to gtypes.Schema for Google providers."""
        schemas = self._rust_generator.generate_tool_schemas(tools)

        # Check if this is a Google provider
        provider_name = self._provider.split(":")[0]
        if provider_name in ("google-gemini", "google-genai", "vertex-gemini"):
            if gtypes is None:
                raise ImportError(
                    "Google GenAI library not available. Install with: pip install google-genai"
                )
            # Convert parameters dict to gtypes.Schema
            for schema in schemas:
                if "function_declarations" in schema:
                    for fd in schema["function_declarations"]:
                        if "parameters" in fd and isinstance(fd["parameters"], dict):
                            fd["parameters"] = _convert_to_google_schema(fd["parameters"])

        return schemas


def get_schema_generator(provider: str) -> SchemaGenerator:
    """Get a schema generator for the given provider."""
    if not supports_native_tools(provider):
        raise ValueError(f"Native tools not supported for provider: {provider}")
    return SchemaGenerator(provider)
