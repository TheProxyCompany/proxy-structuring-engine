import pytest
from pse_core.state_machine import StateMachine

from pse.types.base.phrase import PhraseStateMachine
from pse.types.integer import IntegerStateMachine


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("0", 0),
        ("1", 1),
        ("123", 123),
        ("456789", 456789),
        ("000123", 123),  # Leading zeros should be handled
    ],
)
def test_integer_acceptor_multi_char_advancement(input_string, expected_value):
    """Test IntegerAcceptor with multi-character advancement."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    advanced_steppers = sm.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers), (
        f"IntegerAcceptor should accept input '{input_string}'."
    )
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("7", 7),
        ("89", 89),
        ("123456", 123456),
        ("000789", 789),  # Leading zeros should be handled
    ],
)
def test_integer_acceptor(input_string, expected_value):
    """Test IntegerAcceptor with single-character advancement."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    for char in input_string:
        steppers = sm.advance_all_basic(steppers, char)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"IntegerAcceptor should accept input '{input_string}'."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"


def test_integer_acceptor_invalid_input():
    """Test IntegerAcceptor with invalid inputs."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    invalid_inputs = ["abc", "12a4", "3.14", "-123", "", "@@@", "123abc456"]

    for input_string in invalid_inputs:
        steppers = sm.get_steppers()
        advanced_steppers = sm.advance_all_basic(steppers, input_string)
        assert not any(
            stepper.has_reached_accept_state() for stepper in advanced_steppers
        ), f"Input '{input_string}' should not be accepted."


def test_integer_acceptor_empty_input():
    """Test IntegerAcceptor with empty input."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    input_string = ""

    steppers = sm.get_steppers()
    advanced_steppers = sm.advance_all_basic(steppers, input_string)

    assert not any(
        stepper.has_reached_accept_state() for stepper in advanced_steppers
    ), "Empty input should not be accepted."


def test_integer_acceptor_partial_input():
    """Test IntegerAcceptor with input containing invalid characters."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    input_string = "12a34"

    steppers = sm.get_steppers()
    advanced_steppers = sm.advance_all_basic(steppers, input_string)

    assert not any(
        stepper.has_reached_accept_state() for stepper in advanced_steppers
    ), "Input with invalid characters should not be accepted."


def test_integer_acceptor_in_state_machine_sequence():
    """Test IntegerAcceptor within a StateMachine sequence along with other acceptors."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("Number: "), 1)],
            1: [(integer_acceptor, 2)],
        },
        start_state=0,
        end_states=[2],
    )

    input_string = "Number: 42"

    steppers = sm.get_steppers()
    advanced_steppers = sm.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers), (
        "Combined text and integer input should be accepted."
    )
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            expected_value = "Number: 42"
            assert value == expected_value, (
                f"Expected '{expected_value}', got '{value}'"
            )


def test_integer_acceptor_char_by_char_in_state_machine():
    """Test IntegerAcceptor within a StateMachine sequence, advancing one character at a time."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("Value: "), 1)],
            1: [(integer_acceptor, 2)],
        },
        start_state=0,
        end_states=[2],
    )

    input_string = "Value: 9876"
    steppers = sm.get_steppers()
    for char in input_string:
        steppers = sm.advance_all_basic(steppers, char)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Combined text and integer input should be accepted when advancing char by char."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            expected_value = "Value: 9876"
            assert value == expected_value, (
                f"Expected '{expected_value}', got '{value}'"
            )


def test_integer_acceptor_zero():
    """Test IntegerAcceptor with zero."""
    integer_acceptor = IntegerStateMachine()
    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    input_string = "0"

    steppers = sm.get_steppers()
    advanced_steppers = sm.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers), (
        "Zero should be accepted."
    )
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value == 0, f"Expected 0, got {value}"


def test_integer_acceptor_large_number():
    """Test IntegerAcceptor with a large number."""
    integer_acceptor = IntegerStateMachine()
    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    input_string = "12345678901234567890"

    steppers = sm.get_steppers()
    advanced_steppers = sm.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers), (
        "Large numbers should be accepted."
    )
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert int(value) == 1.2345678901234567e19, (
                f"Expected 12345678901234567890, got {value}"
            )


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("0000", 0),
        ("007", 7),
        ("000000", 0),
        ("000001", 1),
    ],
)
def test_integer_acceptor_leading_zeros(input_string, expected_value):
    """Test IntegerAcceptor handling of leading zeros."""
    integer_acceptor = IntegerStateMachine()

    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    advanced_steppers = sm.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers), (
        f"IntegerAcceptor should accept input '{input_string}'."
    )
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"
