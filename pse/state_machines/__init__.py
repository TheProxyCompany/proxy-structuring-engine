import json
from collections.abc import Callable
from typing import Any

from pse_core.state_machine import StateMachine
from pydantic import BaseModel

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.encapsulated import EncapsulatedStateMachine
from pse.state_machines.composite.wait_for import WaitFor
from pse.state_machines.schema.any_schema import AnySchemaStateMachine
from pse.state_machines.schema.number_schema import NumberSchemaStateMachine
from pse.state_machines.schema.string_schema import StringSchemaStateMachine
from pse.state_machines.types.array import ArrayStateMachine
from pse.state_machines.types.boolean import BooleanStateMachine
from pse.state_machines.types.enum import EnumStateMachine
from pse.state_machines.types.object import ObjectStateMachine
from pse.util.generate_schema import callable_to_json_schema, pydantic_to_json_schema


def get_state_machine(
    schema: dict[str, Any] | type[BaseModel] | Callable[..., Any],
    context: dict[str, Any] | None = None,
    delimiters: tuple[str, str] | None = None,
    buffer_length: int = -1,
) -> StateMachine:
    """
    Create a state_machine to validate input based on the provided schema.

    Args:
        schema (Dict[str, Any]): The schema to validate against.
        context (Optional[Dict[str, Any]]): Contextual information for schema definitions and path.
            Defaults to {"defs": defaultdict(dict), "path": ""}.
        delimiters (Optional[Tuple[str, str]]): The delimiters to use for encapsulation.
        buffer_length (int): The length of the buffer to use for encapsulation.

    Returns:
        StateMachine: An state_machine that validates input based on the schema.
    """
    if context is None:
        context = {"defs": {"#": schema}, "path": ""}

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = pydantic_to_json_schema(schema)
    elif callable(schema):
        schema = callable_to_json_schema(schema)

    state_machine = _get_state_machine(schema, context)
    if delimiters:
        return EncapsulatedStateMachine(
            state_machine,
            delimiters,
            buffer_length,
        )
    elif buffer_length > 0:
        return WaitFor(state_machine, buffer_length)
    else:
        return state_machine


def _get_state_machine(
    schema: dict[str, Any],
    context: dict[str, Any]
) -> StateMachine:

    # handle nullable
    if schema.get("nullable"):
        non_nullable_schema: dict[str, Any] = schema.copy()
        del non_nullable_schema["nullable"]
        return AnySchemaStateMachine([{"type": "null"}, non_nullable_schema], context)

    # handle $defs
    if "$defs" in schema:
        schema_defs: dict[str, Any] = schema["$defs"]
        for def_name, def_schema in schema_defs.items():
            context["defs"][f"#/$defs{context['path']}/{def_name}"] = def_schema
            context["defs"][f"#/$defs/{def_name}"] = def_schema

    # resolve subschemas
    schemas = resolve_subschemas(schema, context["defs"], {})
    if len(schemas) == 1:
        schema = schemas[0]
    else:
        return AnySchemaStateMachine(schemas, context)

    if "not" in schema:
        raise ValueError("The 'not' keyword is not supported due to limitations with autoregressive generation.")

    schema_type: Any | None = schema.get("type")

    if isinstance(schema_type, list):
        merged_schemas: list[dict[str, Any]] = [
            {**schema, "type": type_} for type_ in schema_type
        ]
        return AnySchemaStateMachine(merged_schemas, context)

    # Infer schema type based on properties if not explicitly defined
    if schema_type is None:
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"

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
        from pse.state_machines.schema.object_schema import ObjectSchemaStateMachine
        state_machine = ObjectSchemaStateMachine(schema, context)
    elif schema_type == "array" and "items" in schema:
        from pse.state_machines.schema.array_schema import ArraySchemaStateMachine
        state_machine = ArraySchemaStateMachine(schema, context)
    elif schema_type == "array":
        state_machine = ArrayStateMachine()
    elif schema_type == "object":
        state_machine = ObjectStateMachine()
    else:
        raise ValueError(f"unknown schema type: {schema}")

    return state_machine


def resolve_subschemas(
    schema: dict[str, Any],
    defs: dict[str, Any],
    visited_refs: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Resolve references and combine subschemas within a JSON schema.

    This function processes the JSON schema to resolve any "$ref" references
    and combine schemas using combinators like "allOf", "anyOf", and "oneOf".

    Args:
        schema (Dict[str, Any]): The JSON schema to resolve.
        defs (dict[str, Any]): Definitions available for resolving "$ref" references.
        visited_refs (dict[str, Any]): Tracks visited references to prevent infinite recursion.

    Returns:
        List[Dict[str, Any]]: A list of resolved subschemas.

    Raises:
        DefinitionNotFoundError: If a referenced definition is not found in defs.
    """
    if "$ref" in schema:
        schema_ref: str = schema["$ref"]
        if schema_ref in visited_refs:
            return visited_refs[schema_ref]
        schema_def: dict[str, Any] | None = defs.get(schema_ref)
        if schema_def is None:
            raise ValueError(f"definition not found: {schema_ref}")
        visited_refs[schema_ref] = []
        resolved: list[dict[str, Any]] = resolve_subschemas(schema_def, defs, visited_refs)
        visited_refs[schema_ref].extend(resolved)
        return resolved

    for key in ["allOf", "anyOf", "oneOf"]:
        if key not in schema:
            continue

        base_schema = {k: v for k, v in schema.items() if k != key}
        base_schemas = resolve_subschemas(base_schema, defs, visited_refs)
        combined_schemas = base_schemas if key == "allOf" else []

        for subschema in schema[key]:
            resolved_subschemas = resolve_subschemas(subschema, defs, visited_refs)
            if key == "allOf":
                combined_schemas = [{**ms, **rs} for ms in combined_schemas for rs in resolved_subschemas]
            else:
                combined_schemas.extend([{**ms, **rs} for rs in resolved_subschemas for ms in base_schemas])

        return combined_schemas

    return [schema]
