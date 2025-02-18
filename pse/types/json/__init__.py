from collections.abc import Callable
from typing import Any, TypeAlias

from pse_core.state_machine import StateMachine
from pydantic import BaseModel

from pse.types.array import ArrayStateMachine
from pse.types.base.chain import ChainStateMachine
from pse.types.base.encapsulated import EncapsulatedStateMachine
from pse.types.base.phrase import PhraseStateMachine
from pse.types.base.wait_for import WaitFor
from pse.types.boolean import BooleanStateMachine
from pse.types.enum import EnumStateMachine
from pse.types.json.any_json_schema import AnySchemaStateMachine
from pse.types.json.json_number import NumberSchemaStateMachine
from pse.types.json.json_string import StringSchemaStateMachine
from pse.types.json.json_value import JsonStateMachine
from pse.types.object import ObjectStateMachine

SchemaDefinition: TypeAlias = (
    type[BaseModel] | dict[str, Any] | Callable[..., Any] | str
)
JSONSchemaSource: TypeAlias = SchemaDefinition | list[SchemaDefinition]
"""
The different object types that can be used as a schema in the structuring engine.
"""


def schema_state_machine(
    json_schemable: JSONSchemaSource,
    json_delimiters: tuple[str, str] | None = None,
    min_buffer_length: int = -1,
) -> tuple[dict[str, Any], StateMachine]:
    json_schema = generate_json_schema(json_schemable)
    json_state_machine = json_schema_to_state_machine(json_schema)
    if json_delimiters:
        return (
            json_schema,
            EncapsulatedStateMachine(
                json_state_machine,
                json_delimiters,
                min_buffer_length,
            )
        )
    elif min_buffer_length >= 0:
        return (
            json_schema,
            WaitFor(
                json_state_machine,
                min_buffer_length,
            ),
        )
    else:
        return (json_schema, json_state_machine)


def generate_json_schema(source: JSONSchemaSource) -> dict[str, Any]:
    """
    Convert the given schema into an object that can be used by the engine.

    Args:
        schema: The schema to convert.
        Can be a Pydantic model, a callable, a dictionary, or a list.

    Returns:
        The converted schema.
    """
    from pse.types.json.schema_sources.from_function import callable_to_schema
    from pse.types.json.schema_sources.from_pydantic import pydantic_to_schema

    if isinstance(source, type) and issubclass(source, BaseModel):
        return pydantic_to_schema(source)
    elif callable(source):
        return callable_to_schema(source)
    elif isinstance(source, dict):
        return source["schema"] if "schema" in source else source
    elif isinstance(source, list):
        schemas = []
        for s in source:
            if isinstance(s, type) and issubclass(s, BaseModel):
                schemas.append(pydantic_to_schema(s))
            elif isinstance(s, dict):
                schemas.append(s)
            else:
                raise ValueError(f"Invalid schema: {s}")
        return {"oneOf": schemas}
    else:
        try:
            import json

            return json.loads(source)
        except Exception as e:
            raise ValueError(f"Invalid schema: {source}") from e


def json_schema_to_state_machine(
    schema: dict[str, Any], context: dict[str, Any] | None = None
) -> StateMachine:
    from pse.types.json.json_array import ArraySchemaStateMachine
    from pse.types.json.json_object import ObjectSchemaStateMachine

    if context is None:
        context = {"defs": {"#": schema}, "path": ""}

    # handle nullable
    if schema.get("nullable") and not schema.get("did_handle_nullable"):
        schema["did_handle_nullable"] = True
        return AnySchemaStateMachine([{"type": "null"}, schema], context)

    # handle $defs
    if "$defs" in schema:
        context["defs"].update(
            {
                # Add definitions with both absolute and relative paths
                # This allows references like "#/$defs/foo" and "#/$defs/path/foo"
                f"#/$defs{path}/{name}": def_schema
                for name, def_schema in schema["$defs"].items()
                for path in [context["path"], ""]
            }
        )

    processed_schema = process_json_schema(schema, context["defs"], {})

    if len(processed_schema) > 1:
        return AnySchemaStateMachine(processed_schema, context)
    elif not processed_schema:
        raise ValueError("no schemas found")

    schema = processed_schema[0]
    schema_type = schema.get("type", None)
    if isinstance(schema_type, list):
        merged_schemas: list[dict[str, Any]] = [
            {**schema, "type": type_} for type_ in schema_type
        ]
        return AnySchemaStateMachine(merged_schemas, context)

    if not schema_type:
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"
        else:
            schema_type = "any"

    if schema_type == "boolean":
        state_machine = BooleanStateMachine()
    elif schema_type == "null":
        state_machine = PhraseStateMachine("null", is_optional=True)
    elif schema_type in ["number", "integer"]:
        state_machine = NumberSchemaStateMachine(schema)
    elif schema_type == "string" or "enum" in schema or "const" in schema:
        if "enum" in schema:
            state_machine = EnumStateMachine(schema["enum"])
        elif "const" in schema:
            state_machine = ChainStateMachine(
                [
                    PhraseStateMachine('"'),
                    PhraseStateMachine(schema["const"]),
                    PhraseStateMachine('"'),
                ]
            )
        else:
            state_machine = StringSchemaStateMachine(schema)
    elif schema_type == "object" and "properties" in schema:
        state_machine = ObjectSchemaStateMachine(schema, context)
    elif schema_type == "array" and "items" in schema:
        state_machine = ArraySchemaStateMachine(schema, context)
    elif schema_type == "set":
        schema["uniqueItems"] = True
        state_machine = ArraySchemaStateMachine(schema, context)
    elif schema_type == "array" or schema_type == "tuple":
        state_machine = ArrayStateMachine()
    elif schema_type == "object":
        state_machine = ObjectStateMachine()
    else:
        state_machine = JsonStateMachine()

    return state_machine


def process_json_schema(
    schema: dict[str, Any] | None,
    definitions: dict[str, dict[str, Any]],
    visited: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Resolve references and combine subschemas within a schema.

    Args:
        schema: The schema to resolve.
        definitions: Definitions available for resolving "$ref" references.
        visited: Tracks visited schemas to prevent infinite recursion.

    Returns:
        list[dict[str, Any]]: A list of resolved subschemas.
    """
    if schema is None:
        return []

    if "$ref" in schema:
        schema_reference: str = schema["$ref"]
        if schema_reference in visited:
            return visited[schema_reference]
        else:
            visited[schema_reference] = []

        if schema_reference not in definitions:
            raise ValueError(f"definition not found: {schema_reference}")

        resolved = process_json_schema(
            definitions.get(schema_reference), definitions, visited
        )
        visited[schema_reference] = resolved
        return resolved

    for key in ["allOf", "anyOf", "oneOf"]:
        if key not in schema:
            continue

        base_schema = {k: v for k, v in schema.items() if k != key}
        base_schemas = process_json_schema(base_schema, definitions, visited)
        combined_schemas = base_schemas if key == "allOf" else []

        for subschema in schema[key]:
            resolved_subschemas = process_json_schema(subschema, definitions, visited)
            if key == "allOf":
                combined_schemas = [
                    {**ms, **rs}
                    for ms in combined_schemas
                    for rs in resolved_subschemas
                ]
            else:
                combined_schemas.extend(
                    [{**ms, **rs} for rs in resolved_subschemas for ms in base_schemas]
                )

        return combined_schemas

    return [schema]
