#!/usr/bin/env python3
"""Test script to verify the Optional-based schema generation changes."""

from typing import Optional
from reson.utils.schema_generators import get_schema_generator


def test_function_with_optional(
    required_param: str,
    optional_param: Optional[str],
    default_param: str = "default_value",
) -> str:
    """Test function with different parameter types."""
    return f"{required_param}, {optional_param}, {default_param}"


def main():
    # Test with OpenAI schema generator
    generator = get_schema_generator("openai")
    schemas = generator.generate_tool_schemas(
        {"test_function": test_function_with_optional}
    )

    schema = schemas[0]["function"]
    required_fields = schema["parameters"]["required"]

    print("Generated schema:")
    print(f"Required fields: {required_fields}")
    print(f"All parameters: {list(schema['parameters']['properties'].keys())}")

    # With the new logic:
    # - required_param: str -> required (not Optional)
    # - optional_param: Optional[str] -> not required (is Optional)
    # - default_param: str = "default_value" -> required (not Optional, even with default)

    expected_required = ["required_param", "default_param"]

    if set(required_fields) == set(expected_required):
        print("✅ PASS: Schema generation correctly treats Optional types as optional")
        print("✅ PASS: Parameters with defaults are still required if not Optional")
    else:
        print(
            f"❌ FAIL: Expected required fields {expected_required}, got {required_fields}"
        )


if __name__ == "__main__":
    main()
