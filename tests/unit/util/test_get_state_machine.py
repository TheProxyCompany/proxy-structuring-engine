from collections import defaultdict
from typing import Any

import pytest

from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.schema.any_schema_acceptor import AnySchemaAcceptor
from pse.acceptors.schema.array_schema_acceptor import ArraySchemaAcceptor
from pse.acceptors.schema.enum_schema_acceptor import EnumSchemaAcceptor
from pse.acceptors.schema.number_schema_acceptor import NumberSchemaAcceptor
from pse.acceptors.schema.object_schema_acceptor import ObjectSchemaAcceptor
from pse.acceptors.schema.string_schema_acceptor import StringSchemaAcceptor
from pse.util.get_state_machine import get_state_machine


@pytest.fixture
def context() -> dict[str, Any]:
    """Fixture providing the default context for tests."""
    return {"defs": defaultdict(dict), "path": ""}


@pytest.mark.parametrize(
    "schema, expected_acceptor_cls, acceptor_len",
    [
        ({"type": "number", "minimum": 0}, NumberSchemaAcceptor, None),
        ({"type": "string", "nullable": True}, AnySchemaAcceptor, 2),
        ({"type": ["string", "number"], "minimum": 0}, AnySchemaAcceptor, 2),
        (
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number", "minimum": 0},
                },
                "required": ["name", "age"],
            },
            ObjectSchemaAcceptor,
            None,
        ),
        (
            {"type": "array", "items": {"type": "string"}, "minItems": 1},
            ArraySchemaAcceptor,
            None,
        ),
        ({"const": "fixed_value"}, TextAcceptor, None),
        ({"enum": ["red", "green", "blue"]}, EnumSchemaAcceptor, None),
        ({"allOf": [{"type": "string"}, {"minLength": 5}]}, StringSchemaAcceptor, None),
        ({"oneOf": [{"type": "string"}, {"type": "number"}]}, AnySchemaAcceptor, 2),
    ],
)
def test_get_acceptor_schema_types(
    schema: dict[str, Any],
    expected_acceptor_cls: type[Any],
    acceptor_len: int | None,
    context: dict[str, Any],
) -> None:
    """Test get_json_acceptor with various schema types and expected acceptors."""
    acceptor = get_state_machine(schema, context)
    assert isinstance(
        acceptor, expected_acceptor_cls
    ), f"Expected {expected_acceptor_cls.__name__} for schema {schema}"
    if acceptor_len is not None:
        assert isinstance(acceptor, AnySchemaAcceptor)
        assert (
            len(acceptor.acceptors) == acceptor_len
        ), f"Expected acceptor length {acceptor_len} for schema {schema}"


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
    acceptor = get_state_machine(schema, context_with_definition)
    assert isinstance(
        acceptor,
        ObjectSchemaAcceptor,
    ), "get_json_acceptor should return an ObjectSchemaAcceptor for $ref schemas referencing object definitions."
