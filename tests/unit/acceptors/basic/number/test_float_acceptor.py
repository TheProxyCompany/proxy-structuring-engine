import pytest
from pse.acceptors.basic.number.float_acceptor import FloatAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.acceptors.basic.text_acceptor import TextAcceptor

@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("0.0", 0.0),
        ("123.456", 123.456),
        ("0.123", 0.123),
        ("98765.4321", 98765.4321),
        ("1.0", 1.0),
        ("123.0", 123.0),
        ("9999999999.999999", 9999999999.999999),
    ]
)
def test_float_acceptor_multi_char_advancement(input_string, expected_value):
    """Test FloatAcceptor with multi-character advancement."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))
    print(f"Cursors after advancing: {cursors}")

    assert any(cursor.in_accepted_state() for cursor in cursors), f"FloatAcceptor should accept input '{input_string}'."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            assert value == pytest.approx(expected_value), f"Expected {expected_value}, got {value}"


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("7.89", 7.89),
        ("22.0069", 22.0069),
        ("0.00123", 0.00123),
        ("123456.789", 123456.789),
        ("0.0000", 0.0),
        ("9999.0001", 9999.0001),
    ]
)
def test_float_acceptor_single_char_advancement(input_string, expected_value):
    """Test FloatAcceptor with single-character advancement."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())

    for char in input_string:
        cursors = list(sm.advance_all(cursors, char))
        print(f"Cursors after advancing '{char}': {cursors}")

    assert any(cursor.in_accepted_state() for cursor in cursors), f"FloatAcceptor should accept input '{input_string}'."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            assert value == pytest.approx(expected_value), f"Expected {expected_value}, got {value}"


def test_float_acceptor_invalid_input():
    """Test FloatAcceptor with invalid inputs."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    invalid_inputs = [
        "abc",
        "12a4",
        "3.14.15",
        "-123.456",
        "1..23",
        "--3.14",
        "",
        ".456",  # Starts with a dot but no leading digit
        "123a.456",
        "123.-456",
        "123e456",  # Exponential notation not supported
        ".",        # Just a dot
    ]

    for input_string in invalid_inputs:
        cursors = list(sm.get_cursors())
        cursors = list(sm.advance_all(cursors, input_string))
        print(f"Testing invalid input '{input_string}': Cursors: {cursors}")
        assert not any(cursor.in_accepted_state() for cursor in cursors), f"Input '{input_string}' should not be accepted."


def test_float_acceptor_empty_input():
    """Test FloatAcceptor with empty input."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = ""

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))
    print(f"Cursors after empty input: {cursors}")

    assert not any(cursor.in_accepted_state() for cursor in cursors), "Empty input should not be accepted."


def test_float_acceptor_partial_input():
    """Test FloatAcceptor with input containing invalid characters."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "12.3a4"

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))
    print(f"Cursors after partial invalid input '{input_string}': {cursors}")

    assert not any(cursor.in_accepted_state() for cursor in cursors), "Input with invalid characters should not be accepted."


def test_float_acceptor_in_state_machine_sequence():
    """Test FloatAcceptor within a StateMachine sequence along with other acceptors."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Number: "), 1)],
            1: [(float_acceptor, 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    input_string = "Number: 3.14159"

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))
    print(f"Cursors after advancing with input '{input_string}': {cursors}")

    assert any(cursor.in_accepted_state() for cursor in cursors), "Combined text and float input should be accepted."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            expected_value = "Number: 3.14159"
            assert value == expected_value, f"Expected '{expected_value}', got '{value}'"


def test_float_acceptor_char_by_char_in_state_machine():
    """Test FloatAcceptor within a StateMachine sequence, advancing one character at a time."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Value: "), 1)],
            1: [(float_acceptor, 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    input_string = "Value: 0.0001"
    cursors = list(sm.get_cursors())
    for char in input_string:
        cursors = list(sm.advance_all(cursors, char))
        print(f"Cursors after advancing '{char}': {cursors}")

    assert any(cursor.in_accepted_state() for cursor in cursors), "Combined text and float input should be accepted when advancing char by char."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            expected_value = "Value: 0.0001"
            assert value == expected_value, f"Expected '{expected_value}', got '{value}'"


def test_float_acceptor_zero():
    """Test FloatAcceptor with zero and zero fractions."""
    float_acceptor = FloatAcceptor()
    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    inputs = ["0.0", "0.0000", "123.0000"]
    expected_values = [0.0, 0.0, 123.0]

    for input_string, expected_value in zip(inputs, expected_values):
        cursors = list(sm.get_cursors())
        cursors = list(sm.advance_all(cursors, input_string))
        print(f"Cursors after advancing with '{input_string}': {cursors}")

        assert any(cursor.in_accepted_state() for cursor in cursors), f"Input '{input_string}' should be accepted."
        for cursor in cursors:
            if cursor.in_accepted_state():
                value = cursor.get_value()
                assert value == pytest.approx(expected_value), f"Expected {expected_value}, got {value}"


def test_float_acceptor_large_number():
    """Test FloatAcceptor with a large floating-point number."""
    float_acceptor = FloatAcceptor()
    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "12345678901234567890.123456789"
    expected_value = 1.2345678901234568e+19  # Adjusted for float precision

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))
    print(f"Cursors after advancing large number '{input_string}': {cursors}")

    assert any(cursor.in_accepted_state() for cursor in cursors), "Large floating-point numbers should be accepted."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            assert value == pytest.approx(expected_value), f"Expected {expected_value}, got {value}"
