from collections import defaultdict
from typing import Any

import pytest

from pse.types.base.chain import ChainStateMachine
from pse.types.enum import EnumStateMachine
from pse.types.json import _json_schema_to_state_machine
from pse.types.json.any_json_schema import AnySchemaStateMachine
from pse.types.json.json_array import ArraySchemaStateMachine
from pse.types.json.json_number import NumberSchemaStateMachine
from pse.types.json.json_object import ObjectSchemaStateMachine
from pse.types.json.json_string import StringSchemaStateMachine


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
        ({"type": "string", "const": "fixed_value"}, ChainStateMachine, None),
    ],
)
def test_get_acceptor_schema_types(
    schema: dict[str, Any],
    expected_acceptor_cls: type[Any],
    acceptor_len: int | None,
) -> None:
    """Test get_json_acceptor with various schema types and expected acceptors."""
    state_machine = _json_schema_to_state_machine(schema)
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
    state_machine = _json_schema_to_state_machine(schema, context_with_definition)
    assert isinstance(
        state_machine,
        ObjectSchemaStateMachine,
    ), (
        "get_json_acceptor should return an ObjectSchemaAcceptor for $ref schemas referencing object definitions."
    )
