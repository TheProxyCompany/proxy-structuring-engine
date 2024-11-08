import pytest
from typing import Any
from pse.acceptors.collections.array_acceptor import ArrayAcceptor, ArrayWalker


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
    walkers = list(acceptor.get_walkers())
    for char in json_string:
        walkers = [walker for _, walker in acceptor.advance_all_walkers(walkers, char)]
    if not any(walker.has_reached_accept_state() for walker in walkers):
        raise AssertionError("No walker in accepted state")
    # Assuming the first accepted walker contains the parsed value
    for walker in walkers:
        if walker.has_reached_accept_state():
            return walker.get_current_value()
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


# Tests for walker behavior
def test_walker_initialization(acceptor: ArrayAcceptor):
    """Test that the walker initializes with an empty value list."""
    walker = ArrayWalker(acceptor)
    assert walker.value == []


def test_walker_clone(acceptor: ArrayAcceptor):
    """Test that cloning a walker duplicates its state correctly."""
    walker = ArrayWalker(acceptor)
    walker.value.append(123)
    cloned_walker = walker.clone()
    assert cloned_walker.value == [123]
    cloned_walker.value.append(456)
    assert walker.value != cloned_walker.value


# Parameterized tests for invalid arrays
@pytest.mark.parametrize(
    "json_string",
    [
        "[123, 456",  # Missing closing bracket
        "[123, 456, ]",  # Trailing comma
    ],
)
def test_invalid_arrays(acceptor: ArrayAcceptor, json_string: str):
    """Test that a JSONParsingError is raised for invalid arrays."""
    with pytest.raises(AssertionError):
        parse_array(acceptor, json_string)
