"""OpenAI-format tool schema generator."""

from typing import Dict, Any, Callable, List

from .base import SchemaGenerator


class OpenAISchemaGenerator(SchemaGenerator):
    """Generate OpenAI-format tool schemas."""

    def get_provider_name(self) -> str:
        return "openai"

    def generate_tool_schemas(self, tools: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Generate OpenAI tools format: [{"type": "function", "function": {...}}]"""
        schemas = []
        for tool_name, tool_func in tools.items():
            func_info = self._extract_function_info(tool_func)

            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": func_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": func_info["parameters"],
                        "required": func_info["required"],
                    },
                },
            }
            schemas.append(schema)
        return schemas
