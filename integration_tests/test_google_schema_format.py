#!/usr/bin/env python3
"""Test script to figure out the correct Google Schema format by actually using it."""

import os
import asyncio
import pytest
from google.genai import types
from google import genai


@pytest.mark.asyncio
async def test_schema_with_real_client():
    """Test schemas by actually using them with the Google GenAI client."""

    # Use VertexAI setup like in reson project
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("❌ No GOOGLE_APPLICATION_CREDENTIALS found")
        return

    # Create client with VertexAI setup (need project and location)
    import google.auth

    # Get default credentials and project
    try:
        creds, project_id = google.auth.default()
        client = genai.Client(vertexai=True, project=project_id, location="us-central1")
    except Exception as e:
        print(f"❌ VertexAI client setup failed: {e}")
        return
    print("✅ Google client created")

    # Test 1: Simple properties as dicts with type enums
    print("\n🧪 Test 1: Dict properties with type enums")
    try:
        schema1 = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "location": {
                    "type": types.Type.STRING,
                    "description": "The location to get weather for",
                }
            },
            required=["location"],
        )

        tool1 = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_weather",
                    description="Get weather information",
                    parameters=schema1,
                )
            ]
        )

        # Try to use it
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="What's the weather in Paris?")]
                )
            ],
            config=types.GenerateContentConfig(tools=[tool1]),
        )
        print(response)
        print("✅ Test 1: Dict properties with type enums - SUCCESS!")

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    # Test 2: Properties as Schema objects
    print("\n🧪 Test 2: Properties as Schema objects")
    try:
        schema2 = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "location": types.Schema(
                    type=types.Type.STRING,
                    description="The location to get weather for",
                )
            },
            required=["location"],
        )

        tool2 = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_weather",
                    description="Get weather information",
                    parameters=schema2,
                )
            ]
        )

        # Try to use it
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="What's the weather in Paris?")]
                )
            ],
            config=types.GenerateContentConfig(tools=[tool2]),
        )
        print(response)
        print("✅ Test 2: Properties as Schema objects - SUCCESS!")

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    # Test 3: Our current broken approach - to see exact error
    print("\n🧪 Test 3: What our generator currently produces")
    try:
        from reson.utils.schema_generators import get_schema_generator

        def dummy_tool(location: str) -> str:
            """Get weather for location."""
            return f"Weather in {location}"

        generator = get_schema_generator("vertex-gemini")
        schemas = generator.generate_tool_schemas({"get_weather": dummy_tool})

        print(f"Generated schemas: {schemas}")

        # Try to use the generated schema
        if schemas and "function_declarations" in schemas[0]:
            func_decl = schemas[0]["function_declarations"][0]
            tool3 = types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=func_decl["name"],
                        description=func_decl["description"],
                        parameters=func_decl["parameters"],
                    )
                ]
            )

            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text="What's the weather in Paris?")],
                    )
                ],
                config=types.GenerateContentConfig(tools=[tool3]),
            )
            print(response)
            print("✅ Test 3: Our current generator - SUCCESS!")

    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_schema_with_real_client())
