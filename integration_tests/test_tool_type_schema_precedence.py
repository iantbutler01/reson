import warnings
from typing import List, Dict, Optional, Union as TypingUnion

import pytest

from reson.reson import Runtime
from reson.stores import MemoryStore
from reson.utils.schema_generators import get_schema_generator
from reson.types import Deserializable


# Auxiliary classes to exercise named union aliases and nesting
class Cop(Deserializable):
    badge: str


class Nurse(Deserializable):
    license_id: str


# Named union alias (PEP 604) - tests extra indirection
Worker = Cop | Nurse


class Team(Deserializable):
    name: str
    members: List[str]


class ToolQuery(Deserializable):
    "Run the tool with typed parameters"

    query: str
    limit: int
    worker: Optional[Worker]  # Optional named-union alias
    teams: List[Team]  # Nested array of objects
    metadata: Dict[str, str]  # Dict → additionalProperties schema


def tool_fn(
    query: str,
    limit: int,
    worker: TypingUnion[Cop, Nurse, None],
    teams: List[Team],
    metadata: Dict[str, str],
) -> str:
    "This docstring should be overridden by ToolQuery.__doc__"
    return "ok"


def tool_fn_mismatch(
    query: str,
    limit: int,
    worker: TypingUnion[Cop, Nurse, None],
    teams: List[Team],
    metadata: Dict[str, str],
    extra: int,  # not present in ToolQuery → should warn
) -> str:
    return "ok"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_schema_prefers_tool_type_and_warns_on_union(provider):
    rt = Runtime(model=provider, store=MemoryStore(), native_tools=True)

    # Capture warnings for both named-union collapse and potential mismatches
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Register matching tool first (should not warn about name/type mismatches)
        rt.tool(tool_fn, tool_type=ToolQuery)

        # Register mismatch tool (should warn about extra param)
        rt.tool(tool_fn_mismatch, tool_type=ToolQuery)

        # Build provider schema
        gen = get_schema_generator(provider)
        schemas = gen.generate_tool_schemas(rt._tools)

    # Basic structure check
    assert isinstance(schemas, list) and len(schemas) >= 1

    # Find the schema for tool_fn
    def _find(schema_list, name):
        for s in schema_list:
            if provider in ("openai", "custom-openai", "openrouter"):
                if (
                    s.get("type") == "function"
                    and s.get("function", {}).get("name") == name
                ):
                    return s
            elif provider in ("anthropic", "bedrock", "google-anthropic"):
                if s.get("name") == name:
                    return s
        return None

    s1 = _find(schemas, "tool_fn")
    assert s1 is not None

    # Provider-specific extraction of parameters/description
    if provider == "openai":
        params = s1["function"]["parameters"]
        props = params["properties"]
        required = params["required"]
        desc = s1["function"]["description"]
    elif provider == "anthropic":
        params = s1["input_schema"]
        props = params["properties"]
        required = params["required"]
        desc = s1["description"]
    else:
        raise AssertionError("Unexpected provider in this test")

    # Description should come from ToolQuery.__doc__
    assert "Run the tool with typed parameters" in desc

    # Ensure all ToolQuery fields are present (schema comes from tool_type, not function)
    assert set(props.keys()) >= {"query", "limit", "worker", "teams", "metadata"}
    assert "query" in required  # limit is required in ToolQuery too, unless defaulted

    # metadata should be an object with additionalProperties string
    metadata_schema = props["metadata"]
    assert metadata_schema["type"] == "object"
    assert "additionalProperties" in metadata_schema
    ap = metadata_schema["additionalProperties"]
    assert isinstance(ap, dict) and ap.get("type") == "string"

    # teams should be array of objects with nested properties
    teams_schema = props["teams"]
    assert teams_schema["type"] == "array"
    assert "items" in teams_schema
    team_items = teams_schema["items"]
    assert team_items["type"] == "object"
    assert set(team_items["properties"].keys()) >= {"name", "members"}

    # worker named union alias should have triggered a union collapse warning at schema time
    union_warnings = [
        x for x in w if "[reson.schema] Collapsing Union[" in str(x.message)
    ]
    assert len(union_warnings) >= 1

    # mismatch registration should warn about extra function parameter
    mismatch_warnings = [
        x for x in w if "function has params not in tool_type" in str(x.message)
    ]
    assert any("extra" in str(x.message) for x in mismatch_warnings)


@pytest.mark.asyncio
async def test_google_additional_properties_and_nested_if_available():
    # Only run if google.genai is installed
    try:
        from google.genai import types as gtypes  # type: ignore
    except Exception:
        pytest.skip("google.genai not installed; skipping Google schema test")

    rt = Runtime(model="google-genai", store=MemoryStore(), native_tools=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rt.tool(tool_fn, tool_type=ToolQuery)
        gen = get_schema_generator("google-genai")
        schemas = gen.generate_tool_schemas(rt._tools)

    assert isinstance(schemas, list) and len(schemas) == 1
    fd = schemas[0]["function_declarations"][0]

    assert fd["name"] == "tool_fn"
    assert "Run the tool with typed parameters" in fd["description"]

    params_schema = fd["parameters"]
    assert isinstance(params_schema, gtypes.Schema)
    # Properties should be present
    assert getattr(params_schema, "properties", None) is not None

    # metadata should be object with additional_properties set to a Schema of STRING
    metadata_schema = params_schema.properties["metadata"]  # type: ignore[index]
    assert metadata_schema.type == gtypes.Type.OBJECT
    assert hasattr(metadata_schema, "additional_properties")
    ap = metadata_schema.additional_properties
    # additional_properties may be Schema or bool; here it should be Schema of STRING
    assert isinstance(ap, gtypes.Schema)
    assert ap.type == gtypes.Type.STRING

    # teams should be ARRAY of OBJECT with nested properties
    teams_schema = params_schema.properties["teams"]  # type: ignore[index]
    assert teams_schema.type == gtypes.Type.ARRAY
    assert isinstance(teams_schema.items, gtypes.Schema)
    assert teams_schema.items is not None
    assert teams_schema.items.type == gtypes.Type.OBJECT
    # Nested properties should be present
    assert getattr(teams_schema.items, "properties", None) is not None
    assert "name" in teams_schema.items.properties  # type: ignore[operator]
    assert "members" in teams_schema.items.properties  # type: ignore[operator]

    # Ensure we saw a union collapse warning earlier
    assert any("[reson.schema] Collapsing Union[" in str(x.message) for x in w)
