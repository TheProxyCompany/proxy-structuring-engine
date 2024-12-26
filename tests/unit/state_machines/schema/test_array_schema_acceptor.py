from typing import Any

import pytest

from pse.state_machines.schema.array_schema_acceptor import ArraySchemaAcceptor


@pytest.fixture
def base_context() -> dict[str, Any]:
    """
    Fixture to provide a common base context for all tests.
    """
    return {"defs": {}, "path": "root"}


def parse_array(state_machine: ArraySchemaAcceptor, json_string: str) -> list[Any]:
    """
    Helper function to parse a JSON array string using the ArraySchemaAcceptor.

    Args:
        state_machine (ArraySchemaAcceptor): The schema state_machine to use for parsing.
        json_string (str): The JSON array string to parse.

    Returns:
        List[Any]: The parsed array.

    Raises:
        ValueError: If the JSON array is invalid or does not meet schema constraints.
    """
    walkers = list(state_machine.get_walkers())
    for char in json_string:
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    if not any(walker.has_reached_accept_state() for walker in walkers):
        raise ValueError(
            "Invalid JSON array or schema constraints not met.", len(json_string)
        )
    parsed_value = next(
        walker.get_current_value()
        for walker in walkers
        if walker.has_reached_accept_state()
    )
    return parsed_value


def test_min_items(base_context):
    """
    Test that min_items returns the correct value based on the schema.
    """
    schema = {"type": "array", "items": {"type": "number"}, "minItems": 3}
    state_machine = ArraySchemaAcceptor(schema, base_context)
    assert state_machine.min_items() == 3, "min_items should return 3."


def test_max_items(base_context):
    """
    Test that max_items returns the correct value based on the schema.
    """
    schema = {"type": "array", "items": {"type": "number"}, "maxItems": 5}
    state_machine = ArraySchemaAcceptor(schema, base_context)
    assert state_machine.max_items() == 5, "max_items should return 5."


@pytest.mark.parametrize(
    "schema, json_array, expected",
    [
        (
            {"type": "array", "items": {"type": "number"}, "minItems": 2},
            "[1, 2]",
            [1, 2],
        ),
        (
            {"type": "array", "items": {"type": "number"}, "maxItems": 3},
            "[1, 2, 3]",
            [1, 2, 3],
        ),
        (
            {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
                "maxItems": 3,
            },
            "[1]",
            [1],
        ),
        (
            {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 4,
            },
            "[1, 2, 3]",
            [1, 2, 3],
        ),
    ],
)
def test_array_parsing_success(schema, json_array, expected, base_context):
    """
    Test parsing arrays that meet the schema constraints.
    """
    state_machine = ArraySchemaAcceptor(schema, base_context)
    parsed_array = parse_array(state_machine, json_array)
    assert (
        parsed_array == expected
    ), f"Array {json_array} should be parsed as {expected}."


@pytest.mark.parametrize(
    "schema, json_array",
    [
        ({"type": "array", "items": {"type": "number"}, "minItems": 3}, "[1, 2]"),
        ({"type": "array", "items": {"type": "number"}, "maxItems": 2}, "[1, 2, 3]"),
        (
            {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 5,
            },
            "[1, 2]",
        ),
        (
            {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 3,
            },
            "[1, 2, 3, 4]",
        ),
    ],
)
def test_array_parsing_failure(schema, json_array, base_context):
    """
    Test parsing arrays that do not meet the schema constraints and should raise ValueError.
    """
    state_machine = ArraySchemaAcceptor(schema, base_context)
    with pytest.raises(ValueError):
        parse_array(state_machine, json_array)
