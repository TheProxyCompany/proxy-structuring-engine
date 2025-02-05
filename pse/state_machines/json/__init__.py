from typing import Any

from pse_core.state_machine import StateMachine

from pse.state_machines.base.chain import ChainStateMachine
from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.json.any_json_schema import AnySchemaStateMachine
from pse.state_machines.json.json_number import NumberSchemaStateMachine
from pse.state_machines.json.json_string import StringSchemaStateMachine
from pse.state_machines.json.json_value import JsonStateMachine
from pse.state_machines.types.array import ArrayStateMachine
from pse.state_machines.types.boolean import BooleanStateMachine
from pse.state_machines.types.enum import EnumStateMachine
from pse.state_machines.types.object import ObjectStateMachine


def json_schema_to_state_machine(
    schema: dict[str, Any], context: dict[str, Any] | None = None
) -> StateMachine:
    from pse.state_machines.json.json_array import ArraySchemaStateMachine
    from pse.state_machines.json.json_object import ObjectSchemaStateMachine

    if context is None:
        context = {"defs": {"#": schema}, "path": ""}

    # handle nullable
    if schema.get("nullable"):
        del schema["nullable"]
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
        schema (dict[str, Any]): The schema to resolve.
        definitions (dict[str, dict[str, Any]]): Definitions available for resolving "$ref" references.
        visited (dict[str, list[dict[str, Any]]]): Tracks visited schemas to prevent infinite recursion.

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
