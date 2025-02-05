from typing import Any

import pytest

from pse.types.array import ArrayStateMachine


@pytest.fixture
def state_machine():
    """Fixture that provides an ArrayAcceptor instance."""
    return ArrayStateMachine()


def parse_array(state_machine: ArrayStateMachine, json_string: str) -> list[Any]:
    """
    Helper function to parse a JSON array string using the ArrayAcceptor.

    Args:
        state_machine (ArrayAcceptor): The ArrayAcceptor instance.
        json_string (str): The JSON array string to parse.

    Returns:
        list[Any]: The parsed array.

    Raises:
        AssertionError: If the JSON array is invalid.
    """
    steppers = state_machine.get_steppers()
    for char in json_string:
        steppers = state_machine.advance_all_basic(steppers, char)
    if not any(stepper.has_reached_accept_state() for stepper in steppers):
        raise AssertionError("No stepper in accepted state")
    # Assuming the first accepted stepper contains the parsed value
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            return stepper.get_current_value()
    return []


# Parameterized tests for valid arrays
@pytest.mark.parametrize(
    "json_string, expected",
    [
        ("[]", []),
        ("[1]", [1]),
        ("[123]", [123]),
        ('[123, 456, "789"]', [123, 456, "789"]),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
    ],
)
def test_valid_arrays(
    state_machine: ArrayStateMachine, json_string: str, expected: list[Any]
):
    """Test parsing of valid JSON arrays."""
    assert parse_array(state_machine, json_string) == expected


# Parameterized tests for invalid arrays
@pytest.mark.parametrize(
    "json_string",
    [
        "[123, 456",  # Missing closing bracket
        "[123, 456, ]",  # Trailing comma
    ],
)
def test_invalid_arrays(state_machine: ArrayStateMachine, json_string: str):
    """Test that an AssertionError is raised for invalid arrays."""
    with pytest.raises(AssertionError):
        parse_array(state_machine, json_string)
