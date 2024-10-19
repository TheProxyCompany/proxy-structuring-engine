import pytest
from pse.schema_acceptors.array_schema_acceptor import ArraySchemaAcceptor
from pse.util.errors import JSONParsingError
from typing import Any, Dict, List

@pytest.fixture
def base_context() -> Dict[str, Any]:
    """
    Fixture to provide a common base context for all tests.
    """
    return {
        "defs": {},
        "path": "root"
    }

def parse_array(acceptor: ArraySchemaAcceptor, json_string: str) -> List[Any]:
    """
    Helper function to parse a JSON array string using the ArraySchemaAcceptor.

    Args:
        acceptor (ArraySchemaAcceptor): The schema acceptor to use for parsing.
        json_string (str): The JSON array string to parse.

    Returns:
        List[Any]: The parsed array.

    Raises:
        JSONParsingError: If the JSON array is invalid or does not meet schema constraints.
    """
    cursors = list(acceptor.get_cursors())
    for char in json_string:
        cursors = list(acceptor.advance_all(cursors, char))
    if not any(cursor.in_accepted_state() for cursor in cursors):
        raise JSONParsingError("Invalid JSON array or schema constraints not met.", len(json_string))
    parsed_value = next(cursor.get_value() for cursor in cursors if cursor.in_accepted_state())
    return parsed_value

def test_min_items(base_context):
    """
    Test that min_items returns the correct value based on the schema.
    """
    schema = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3
    }
    acceptor = ArraySchemaAcceptor(schema, base_context)
    assert acceptor.min_items() == 3, "min_items should return 3."

def test_max_items(base_context):
    """
    Test that max_items returns the correct value based on the schema.
    """
    schema = {
        "type": "array",
        "items": {"type": "number"},
        "maxItems": 5
    }
    acceptor = ArraySchemaAcceptor(schema, base_context)
    assert acceptor.max_items() == 5, "max_items should return 5."

@pytest.mark.parametrize(
    "schema, json_array, expected",
    [
        (
            {"type": "array", "items": {"type": "number"}, "minItems": 2},
            "[1, 2]",
            [1, 2]
        ),
        (
            {"type": "array", "items": {"type": "number"}, "maxItems": 3},
            "[1, 2, 3]",
            [1, 2, 3]
        ),
        (
            {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 3},
            "[1]",
            [1]
        ),
        (
            {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 4},
            "[1, 2, 3]",
            [1, 2, 3]
        ),
    ]
)
def test_array_parsing_success(schema, json_array, expected, base_context):
    """
    Test parsing arrays that meet the schema constraints.
    """
    acceptor = ArraySchemaAcceptor(schema, base_context)
    parsed_array = parse_array(acceptor, json_array)
    assert parsed_array == expected, f"Array {json_array} should be parsed as {expected}."

@pytest.mark.parametrize(
    "schema, json_array",
    [
        (
            {"type": "array", "items": {"type": "number"}, "minItems": 3},
            "[1, 2]"
        ),
        (
            {"type": "array", "items": {"type": "number"}, "maxItems": 2},
            "[1, 2, 3]"
        ),
        (
            {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 5},
            "[1, 2]"
        ),
        (
            {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3},
            "[1, 2, 3, 4]"
        ),
    ]
)
def test_array_parsing_failure(schema, json_array, base_context):
    """
    Test parsing arrays that do not meet the schema constraints and should raise JSONParsingError.
    """
    acceptor = ArraySchemaAcceptor(schema, base_context)
    with pytest.raises(JSONParsingError):
        parse_array(acceptor, json_array)
