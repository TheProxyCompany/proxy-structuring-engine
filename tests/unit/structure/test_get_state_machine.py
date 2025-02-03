from collections import defaultdict
from typing import Any

import pytest

from pse.state_machines import build_state_machine
from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.schema.any_schema import AnySchemaStateMachine
from pse.state_machines.schema.array_schema import ArraySchemaStateMachine
from pse.state_machines.schema.number_schema import NumberSchemaStateMachine
from pse.state_machines.schema.object_schema import ObjectSchemaStateMachine
from pse.state_machines.schema.string_schema import StringSchemaStateMachine
from pse.state_machines.types.enum import EnumStateMachine


@pytest.fixture
def context() -> dict[str, Any]:
    """Fixture providing the default context for tests."""
    return {"defs": defaultdict(dict), "path": ""}


@pytest.mark.parametrize(
    "schema, expected_acceptor_cls, acceptor_len",
    [
        ({"type": "number", "minimum": 0}, NumberSchemaStateMachine, None),
        ({"type": "string", "nullable": True}, AnySchemaStateMachine, 2),
        ({"type": ["string", "number"], "minimum": 0}, AnySchemaStateMachine, 2),
        (
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number", "minimum": 0},
                },
                "required": ["name", "age"],
            },
            ObjectSchemaStateMachine,
            None,
        ),
        (
            {"type": "array", "items": {"type": "string"}, "minItems": 1},
            ArraySchemaStateMachine,
            None,
        ),
        ({"enum": ["red", "green", "blue"]}, EnumStateMachine, None),
        (
            {"allOf": [{"type": "string"}, {"minLength": 5}]},
            StringSchemaStateMachine,
            None,
        ),
        ({"oneOf": [{"type": "string"}, {"type": "number"}]}, AnySchemaStateMachine, 2),
        ({"type": "string", "const": "fixed_value"}, PhraseStateMachine, None),
    ],
)
def test_get_acceptor_schema_types(
    schema: dict[str, Any],
    expected_acceptor_cls: type[Any],
    acceptor_len: int | None,
    context: dict[str, Any],
) -> None:
    """Test get_json_acceptor with various schema types and expected acceptors."""
    state_machine = build_state_machine(schema, context)
    assert isinstance(state_machine, expected_acceptor_cls), (
        f"Expected {expected_acceptor_cls.__name__} for schema {schema}"
    )
    if acceptor_len is not None:
        assert isinstance(state_machine, AnySchemaStateMachine)
        assert len(state_machine.state_machines) == acceptor_len, (
            f"Expected state_machine length {acceptor_len} for schema {schema}"
        )


@pytest.fixture
def context_with_definition() -> dict[str, Any]:
    """Fixture providing context with predefined definitions."""
    context = {"defs": defaultdict(dict), "path": ""}
    context["defs"]["#/definitions/address"] = {
        "type": "object",
        "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
        "required": ["street", "city"],
    }
    return context


def test_get_acceptor_with_ref_schema(context_with_definition: dict[str, Any]) -> None:
    """Test get_json_acceptor with a $ref schema referencing a definition."""
    schema = {"$ref": "#/definitions/address"}
    state_machine = build_state_machine(schema, context_with_definition)
    assert isinstance(
        state_machine,
        ObjectSchemaStateMachine,
    ), (
        "get_json_acceptor should return an ObjectSchemaAcceptor for $ref schemas referencing object definitions."
    )
