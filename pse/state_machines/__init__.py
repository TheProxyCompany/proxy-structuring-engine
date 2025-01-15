import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from pse_core.state_machine import StateMachine

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.schema.any_schema import AnySchemaStateMachine
from pse.state_machines.schema.number_schema import NumberSchemaStateMachine
from pse.state_machines.schema.string_schema import StringSchemaStateMachine
from pse.state_machines.types.array import ArrayStateMachine
from pse.state_machines.types.boolean import BooleanStateMachine
from pse.state_machines.types.enum import EnumStateMachine
from pse.state_machines.types.object import ObjectStateMachine


def get_state_machine(
    schema: dict[str, Any],
    context: dict[str, Any] | None = None,
    start_hook: Callable | None = None,
    end_hook: Callable | None = None,
) -> StateMachine:
    """
    Create an state_machine to validate JSON input based on the provided schema.

    This function initializes a StateMachineAcceptor that enforces the constraints
    defined by the JSON schema. It handles various schema keywords and ensures that
    the input conforms to the expected structure and types.

    Args:
        schema (Dict[str, Any]): The JSON schema to validate against.
        context (Optional[Dict[str, Any]]): Contextual information for schema definitions and path.
            Defaults to {"defs": defaultdict(dict), "path": ""}.
        start_hook (Optional[Callable]): A callable to execute at the start of acceptance.
            Defaults to None.
        end_hook (Optional[Callable]): A callable to execute at the end of acceptance.
            Defaults to None.

    Returns:
        StateMachineAcceptor: An state_machine that validates JSON input based on the schema.

    Raises:
        SchemaNotImplementedError: If the schema contains unsupported keywords like "$id" or "not".
        UnknownSchemaTypeError: If the schema type is unrecognized.
        DefinitionNotFoundError: If a referenced definition is not found in the context.
    """
    if context is None:
        context = {"defs": defaultdict(dict), "path": ""}

        context["defs"]["#"] = schema

    if schema.get("nullable"):
        non_nullable_schema: dict[str, Any] = schema.copy()
        del non_nullable_schema["nullable"]
        return AnySchemaStateMachine([{"type": "null"}, non_nullable_schema], context)

    if "$defs" in schema:
        schema_defs: dict[str, Any] = schema["$defs"]
        if "$id" in schema_defs:
            raise ValueError("$defs.$id is not supported")
        for def_name, def_schema in schema_defs.items():
            # Handle both relative and absolute definition paths
            context["defs"][f"#/$defs{context['path']}/{def_name}"] = def_schema
            context["defs"][f"#/$defs/{def_name}"] = def_schema

    schemas: list[dict[str, Any]] = resolve_subschemas(schema, context["defs"], {})
    if len(schemas) == 1:
        schema = schemas[0]
    else:
        return AnySchemaStateMachine(schemas, context)

    if "not" in schema:
        raise ValueError(
            "The 'not' keyword is not supported due to limitations with autoregressive generation."
        )

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

    # Mapping schema types to their corresponding acceptors
    if schema_type == "boolean":
        state_machine = BooleanStateMachine()
    elif schema_type == "null":
        state_machine = PhraseStateMachine("null")
    elif schema_type in ["number", "integer"]:
        state_machine = NumberSchemaStateMachine(schema)
    elif "enum" in schema:
        state_machine = EnumStateMachine(schema["enum"])
    elif "const" in schema:
        state_machine = PhraseStateMachine(json.dumps(schema["const"]))
    elif schema_type == "string":
        state_machine = StringSchemaStateMachine(schema, start_hook, end_hook)
    elif schema_type == "object":
        if "properties" in schema:
            # Only allows named properties in the object.
            from pse.state_machines.schema.object_schema import (
                ObjectSchemaStateMachine,
            )

            state_machine = ObjectSchemaStateMachine(
                schema, context, start_hook, end_hook
            )
        else:
            # Allows any properties in the object.
            state_machine = ObjectStateMachine()
    elif schema_type == "array":
        from pse.state_machines.schema.array_schema import ArraySchemaStateMachine

        if "items" in schema:
            state_machine = ArraySchemaStateMachine(schema, context)
        else:
            state_machine = ArrayStateMachine()
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
        resolved: list[dict[str, Any]] = resolve_subschemas(
            schema_def, defs, visited_refs
        )
        visited_refs[schema_ref].extend(resolved)
        return resolved

    if "allOf" in schema:
        base_schema: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
        schemas: list[dict[str, Any]] = resolve_subschemas(
            base_schema, defs, visited_refs
        )
        for subschema in schema["allOf"]:
            resolved_subschemas: list[dict[str, Any]] = resolve_subschemas(
                subschema, defs, visited_refs
            )
            schemas = [{**ms, **rs} for ms in schemas for rs in resolved_subschemas]
        return schemas

    if "anyOf" in schema or "oneOf" in schema:
        key: str = "anyOf" if "anyOf" in schema else "oneOf"
        base_schema: dict[str, Any] = {k: v for k, v in schema.items() if k != key}
        base_schemas: list[dict[str, Any]] = resolve_subschemas(
            base_schema, defs, visited_refs
        )
        combined_schemas: list[dict[str, Any]] = []
        for subschema in schema[key]:
            resolved_subschemas: list[dict[str, Any]] = resolve_subschemas(
                subschema, defs, visited_refs
            )
            combined_schemas.extend(
                [{**ms, **rs} for rs in resolved_subschemas for ms in base_schemas]
            )
        return combined_schemas

    return [schema]
