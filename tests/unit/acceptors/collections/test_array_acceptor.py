import pytest
from typing import Any
from pse.acceptors.collections.array_acceptor import ArrayAcceptor

@pytest.fixture
def acceptor():
    """Fixture that provides an ArrayAcceptor instance."""
    return ArrayAcceptor()

def parse_array(acceptor: ArrayAcceptor, json_string: str) -> list[Any]:
    """
    Helper function to parse a JSON array string using the ArrayAcceptor.

    Args:
        acceptor (ArrayAcceptor): The ArrayAcceptor instance.
        json_string (str): The JSON array string to parse.

    Returns:
        list[Any]: The parsed array.

    Raises:
        JSONParsingError: If the JSON array is invalid.
    """
    cursors = list(acceptor.get_cursors())
    for char in json_string:
        cursors = list(acceptor.advance_all(cursors, char))
    if not any(cursor.in_accepted_state() for cursor in cursors):
        raise AssertionError("No cursor in accepted state")
    # Assuming the first accepted cursor contains the parsed value
    for cursor in cursors:
        if cursor.in_accepted_state():
            return cursor.get_value()
    return []

# Parameterized tests for valid arrays
@pytest.mark.parametrize(
    "json_string, expected",
    [
        ("[]", []),
        ("[123]", [123]),
        ('[123, 456, "789"]', [123, 456, 789]),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
    ],
)
def test_valid_arrays(acceptor: ArrayAcceptor, json_string: str, expected: list[Any]):
    """Test parsing of valid JSON arrays."""
    assert parse_array(acceptor, json_string) == expected

# Tests for cursor behavior
def test_cursor_initialization(acceptor: ArrayAcceptor):
    """Test that the cursor initializes with an empty value list."""
    cursor = acceptor.Cursor(acceptor)
    assert cursor.value == []

def test_cursor_clone(acceptor: ArrayAcceptor):
    """Test that cloning a cursor duplicates its state correctly."""
    cursor = acceptor.Cursor(acceptor)
    cursor.value.append(123)
    cloned_cursor = cursor.clone()
    assert cloned_cursor.value == [123]
    cloned_cursor.value.append(456)
    assert cursor.value != cloned_cursor.value

# Parameterized tests for invalid arrays
@pytest.mark.parametrize(
    "json_string",
    [
        "[123, 456",    # Missing closing bracket
        "[123, 456, ]", # Trailing comma
    ],
)
def test_invalid_arrays(acceptor: ArrayAcceptor, json_string: str):
    """Test that a JSONParsingError is raised for invalid arrays."""
    with pytest.raises(AssertionError):
        parse_array(acceptor, json_string)
