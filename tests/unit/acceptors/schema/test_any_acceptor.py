from collections import defaultdict
from typing import Any

import pytest

from pse.state_machines.schema.any_schema_acceptor import AnySchemaAcceptor


@pytest.fixture
def context():
    """Provide the context fixture."""
    return {"defs": defaultdict(dict), "path": ""}


def parse_input(state_machine: AnySchemaAcceptor, json_string: str) -> Any:
    """
    Helper function to parse a JSON string using the AnyOfAcceptor.

    Args:
        state_machine (AnyOfAcceptor): The state_machine instance to use for parsing.
        json_string (str): The JSON string to parse.

    Returns:
        Any: The parsed JSON value.

    Raises:
        JSONParsingError: If the JSON input is invalid or does not match any schema.
    """
    walkers = state_machine.get_walkers()
    walkers = state_machine.advance_all(walkers, json_string)
    for _, walker in walkers:
        if walker.has_reached_accept_state():
            return walker.current_value

    raise ValueError(f"Invalid JSON input for AnyOfAcceptor: {json_string}")


@pytest.mark.parametrize(
    "schemas, token, expected_result",
    [
        (
            [{"type": "number", "minimum": 0}, {"type": "string", "maxLength": 5}],
            "10",
            10,
        ),
        (
            [{"type": "number", "minimum": 0}, {"type": "string", "maxLength": 5}],
            '"test"',
            "test",
        ),
    ],
)
def test_accept_input_matching_single_schema(context, schemas, token, expected_result):
    """Test that input matching a single schema is accepted."""
    state_machine = AnySchemaAcceptor(schemas=schemas, context=context)
    result = parse_input(state_machine, token)
    assert (
        result == expected_result
    ), f"AnyOfAcceptor should accept valid input {token}."


def test_accept_input_matching_multiple_schemas(context):
    """Test that input matching multiple schemas is accepted."""
    # Define overlapping schemas
    schema1 = {"type": "number", "minimum": 0, "maximum": 100}
    schema2 = {"type": "number", "multipleOf": 5}
    state_machine = AnySchemaAcceptor(schemas=[schema1, schema2], context=context)

    valid_input = "25"  # Matches both schemas
    result = parse_input(state_machine, valid_input)
    assert result == 25, "AnyOfAcceptor should accept input matching multiple schemas."


def test_reject_input_not_matching_any_schema(context):
    """Test that input not matching any schema is rejected."""
    schema1 = {"type": "boolean"}
    schema2 = {"type": "null"}
    state_machine = AnySchemaAcceptor(schemas=[schema1, schema2], context=context)

    invalid_input_number = "1"
    invalid_input_string = '"test"'

    # Test with invalid number
    with pytest.raises(ValueError):
        parse_input(state_machine, invalid_input_number)

    # Test with invalid string
    with pytest.raises(ValueError):
        parse_input(state_machine, invalid_input_string)


def test_invalid_schema_initialization(context):
    """Test that initializing AnyOfAcceptor with invalid schemas raises an error."""
    invalid_schema = {"invalid": "schema"}

    with pytest.raises(ValueError):
        AnySchemaAcceptor(schemas=[invalid_schema], context=context)


def test_complex_nested_schemas(context):
    """Test AnyOfAcceptor with complex nested schemas."""
    schema1 = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number", "minimum": 0},
        },
        "required": ["name", "age"],
    }
    schema2 = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    state_machine = AnySchemaAcceptor(schemas=[schema1, schema2], context=context)

    valid_object = '{"name": "Alice", "age": 30}'
    valid_array = '["apple", "banana"]'

    # Test with valid object
    result_object = parse_input(state_machine, valid_object)
    assert result_object == {
        "name": "Alice",
        "age": 30,
    }, "AnyOfAcceptor should accept valid object input."

    # Test with valid array
    result_array = parse_input(state_machine, valid_array)
    assert result_array == [
        "apple",
        "banana",
    ], "AnyOfAcceptor should accept valid array input."


def test_partial_input(context):
    """Test that partial input does not result in acceptance."""
    schema = {"type": "string", "minLength": 5}
    state_machine = AnySchemaAcceptor(schemas=[schema], context=context)

    partial_input = '"test"'

    with pytest.raises(ValueError):
        parse_input(state_machine, partial_input)


@pytest.mark.parametrize(
    "token, expected_result",
    [
        ('"test"', "test"),  # Matches string schema
        ("123", 123),  # Matches number schema
    ],
)
def test_multiple_accepted_walkers(context, token, expected_result):
    """Test that AnyOfAcceptor handles multiple accepted walkers correctly."""
    schema1 = {"type": "string"}
    schema2 = {"type": "number"}
    state_machine = AnySchemaAcceptor(schemas=[schema1, schema2], context=context)

    result = parse_input(state_machine, token)
    assert result == expected_result, f"AnyOfAcceptor should accept input {token}."
