import pytest
from pse_core.state_machine import StateMachine

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.types.integer import IntegerStateMachine


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

    walkers = sm.get_walkers()
    advanced_walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    ), f"IntegerAcceptor should accept input '{input_string}'."
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
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

    walkers = sm.get_walkers()
    for char in input_string:
        walkers = [walker for _, walker in sm.advance_all(walkers, char)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"IntegerAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
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
        walkers = sm.get_walkers()
        advanced_walkers = [
            walker for _, walker in sm.advance_all(walkers, input_string)
        ]
        assert not any(
            walker.has_reached_accept_state() for walker in advanced_walkers
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

    walkers = sm.get_walkers()
    advanced_walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert not any(
        walker.has_reached_accept_state() for walker in advanced_walkers
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

    walkers = sm.get_walkers()
    advanced_walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert not any(
        walker.has_reached_accept_state() for walker in advanced_walkers
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

    walkers = sm.get_walkers()
    advanced_walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    ), "Combined text and integer input should be accepted."
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            expected_value = "Number: 42"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


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
    walkers = sm.get_walkers()
    for char in input_string:
        walkers = [walker for _, walker in sm.advance_all(walkers, char)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and integer input should be accepted when advancing char by char."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            expected_value = "Value: 9876"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


def test_integer_acceptor_zero():
    """Test IntegerAcceptor with zero."""
    integer_acceptor = IntegerStateMachine()
    sm = StateMachine(
        state_graph={0: [(integer_acceptor, 1)]},
        start_state=0,
        end_states=[1],
    )

    input_string = "0"

    walkers = sm.get_walkers()
    advanced_walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    ), "Zero should be accepted."
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
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

    walkers = sm.get_walkers()
    advanced_walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    ), "Large numbers should be accepted."
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            assert (
                int(value) == 1.2345678901234567e19
            ), f"Expected 12345678901234567890, got {value}"


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

    walkers = sm.get_walkers()
    advanced_walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    ), f"IntegerAcceptor should accept input '{input_string}'."
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"
