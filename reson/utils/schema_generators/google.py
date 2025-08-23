"""Google Vertex AI function declaration format schema generator."""

from typing import Dict, Any, Callable, List

try:
    from google.genai import types
except ImportError:
    types = None

from .base import SchemaGenerator


class GoogleSchemaGenerator(SchemaGenerator):
    """Generate Google Vertex AI function declaration format."""

    def get_provider_name(self) -> str:
        return "google"

    def generate_tool_schemas(self, tools: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Generate Google format: [{"function_declarations": [...]}]"""
        function_declarations = []
        for tool_name, tool_func in tools.items():
            func_info = self._extract_function_info(tool_func)

            # Convert to Google's Schema format
            parameters = self._convert_to_google_schema(
                func_info["parameters"], func_info["required"]
            )

            declaration = {
                "name": tool_name,
                "description": func_info["description"],
                "parameters": parameters,
            }
            function_declarations.append(declaration)

        return [{"function_declarations": function_declarations}]

    def _convert_to_google_schema(self, properties: Dict, required: List[str]) -> Any:
        """Convert OpenAPI schema to Google Schema object."""
        if types is None:
            raise ImportError(
                "Google GenAI library not available. Install with: pip install google-genai"
            )

        google_properties = {}

        for prop_name, prop_schema in properties.items():
            google_prop_kwargs = {}

            # Map type names to Google format and create Schema objects
            if prop_schema.get("type") == "integer":
                google_prop_kwargs["type"] = types.Type.INTEGER
            elif prop_schema.get("type") == "number":
                google_prop_kwargs["type"] = types.Type.NUMBER
            elif prop_schema.get("type") == "string":
                google_prop_kwargs["type"] = types.Type.STRING
            elif prop_schema.get("type") == "boolean":
                google_prop_kwargs["type"] = types.Type.BOOLEAN
            elif prop_schema.get("type") == "array":
                google_prop_kwargs["type"] = types.Type.ARRAY
                if "items" in prop_schema:
                    # Create Schema object for array items
                    item_schema = self._convert_single_property_to_google_schema(
                        prop_schema["items"]
                    )
                    google_prop_kwargs["items"] = item_schema
            elif prop_schema.get("type") == "object":
                google_prop_kwargs["type"] = types.Type.OBJECT
                if "properties" in prop_schema:
                    # Recursively convert nested properties
                    nested_properties = {}
                    for nested_prop_name, nested_prop_schema in prop_schema[
                        "properties"
                    ].items():
                        nested_properties[nested_prop_name] = (
                            self._convert_single_property_to_google_schema(
                                nested_prop_schema
                            )
                        )
                    google_prop_kwargs["properties"] = nested_properties
                    google_prop_kwargs["required"] = prop_schema.get("required", [])

            if "description" in prop_schema:
                google_prop_kwargs["description"] = prop_schema["description"]

            # Create actual Schema object
            google_properties[prop_name] = types.Schema(**google_prop_kwargs)

        # Return the main Schema object
        return types.Schema(
            type=types.Type.OBJECT, properties=google_properties, required=required
        )

    def _convert_single_property_to_google_schema(self, prop_schema: Dict) -> Any:
        """Convert a single property schema to Google Schema object."""
        if types is None:
            raise ImportError("Google GenAI library not available")

        google_prop_kwargs = {}

        # Map type names to Google format
        if prop_schema.get("type") == "integer":
            google_prop_kwargs["type"] = types.Type.INTEGER
        elif prop_schema.get("type") == "number":
            google_prop_kwargs["type"] = types.Type.NUMBER
        elif prop_schema.get("type") == "string":
            google_prop_kwargs["type"] = types.Type.STRING
        elif prop_schema.get("type") == "boolean":
            google_prop_kwargs["type"] = types.Type.BOOLEAN
        elif prop_schema.get("type") == "array":
            google_prop_kwargs["type"] = types.Type.ARRAY
            if "items" in prop_schema:
                item_schema = self._convert_single_property_to_google_schema(
                    prop_schema["items"]
                )
                google_prop_kwargs["items"] = item_schema
        elif prop_schema.get("type") == "object":
            google_prop_kwargs["type"] = types.Type.OBJECT
            if "properties" in prop_schema:
                nested_properties = {}
                for nested_prop_name, nested_prop_schema in prop_schema[
                    "properties"
                ].items():
                    nested_properties[nested_prop_name] = (
                        self._convert_single_property_to_google_schema(
                            nested_prop_schema
                        )
                    )
                google_prop_kwargs["properties"] = nested_properties
                google_prop_kwargs["required"] = prop_schema.get("required", [])

        if "description" in prop_schema:
            google_prop_kwargs["description"] = prop_schema["description"]

        return types.Schema(**google_prop_kwargs)
