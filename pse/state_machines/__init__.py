import json
import logging
from typing import Any

from pse_core.state_machine import StateMachine

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.encapsulated import EncapsulatedStateMachine
from pse.state_machines.composite.wait_for import WaitFor
from pse.state_machines.schema.any_schema import AnySchemaStateMachine
from pse.state_machines.schema.number_schema import NumberSchemaStateMachine
from pse.state_machines.schema.string_schema import StringSchemaStateMachine
from pse.state_machines.types.array import ArrayStateMachine
from pse.state_machines.types.boolean import BooleanStateMachine
from pse.state_machines.types.enum import EnumStateMachine
from pse.state_machines.types.json import JsonStateMachine
from pse.state_machines.types.object import ObjectStateMachine
from pse.structure import SchemaType, get_schema_dict

logger = logging.getLogger(__name__)


def build_state_machine(
    schema: SchemaType,
    context: dict[str, Any] | None = None,
    delimiters: tuple[str, str] | None = None,
    min_buffer_length: int = -1,
) -> StateMachine:
    """
    Build a state_machine based on the provided schema.

    Args:
        structure (SchemaType): The schema to validate against.
        context (dict[str, Any] | None): Contextual information for schema definitions and path.
        delimiters (tuple[str, str] | None): The delimiters to indicate the start and end of the schema.
        buffer_length (int): The buffer size before enforcing the schema.

    Returns:
        StateMachine: An state_machine based on the schema.
    """
    if context is None:
        context = {"defs": {"#": schema}, "path": ""}

    structured_schema = get_schema_dict(schema)
    state_machine = schema_to_state_machine(structured_schema, context)
    if delimiters:
        return EncapsulatedStateMachine(
            state_machine,
            delimiters,
            min_buffer_length,
        )
    elif min_buffer_length >= 0:
        return WaitFor(state_machine, min_buffer_length)
    else:
        return state_machine


def schema_to_state_machine(
    schema: dict[str, Any], context: dict[str, Any]
) -> StateMachine:
    from pse.state_machines.schema.array_schema import ArraySchemaStateMachine
    from pse.state_machines.schema.object_schema import ObjectSchemaStateMachine

    # handle nullable
    if schema.get("nullable"):
        del schema["nullable"]
        return AnySchemaStateMachine([{"type": "null"}, schema], context)

    # handle $defs
    if "$defs" in schema:
        schema_defs: dict[str, Any] = schema["$defs"]
        for def_name, def_schema in schema_defs.items():
            context["defs"][f"#/$defs{context['path']}/{def_name}"] = def_schema
            context["defs"][f"#/$defs/{def_name}"] = def_schema

    processed_schema = process_schema(schema, context["defs"], {})

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
            state_machine = PhraseStateMachine(json.dumps(schema["const"]))
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


def process_schema(
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

        resolved = process_schema(
            definitions.get(schema_reference), definitions, visited
        )
        visited[schema_reference] = resolved
        return resolved

    for key in ["allOf", "anyOf", "oneOf"]:
        if key not in schema:
            continue

        base_schema = {k: v for k, v in schema.items() if k != key}
        base_schemas = process_schema(base_schema, definitions, visited)
        combined_schemas = base_schemas if key == "allOf" else []

        for subschema in schema[key]:
            resolved_subschemas = process_schema(subschema, definitions, visited)
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
