import pytest
from pse.acceptors.basic.number.integer_acceptor import IntegerAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.acceptors.basic.text_acceptor import TextAcceptor

@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("0", 0),
        ("1", 1),
        ("123", 123),
        ("456789", 456789),
        ("000123", 123),  # Leading zeros should be handled
    ]
)
def test_integer_acceptor_multi_char_advancement(input_string, expected_value):
    """Test IntegerAcceptor with multi-character advancement."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    print(f"cursors: {cursors}")
    cursors = list(sm.advance_all(cursors, input_string))
    print(f"cursors: {cursors}")

    assert any(cursor.in_accepted_state() for cursor in cursors), f"IntegerAcceptor should accept input '{input_string}'."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("7", 7),
        ("89", 89),
        ("123456", 123456),
        ("000789", 789),  # Leading zeros should be handled
    ]
)
def test_integer_acceptor_single_char_advancement(input_string, expected_value):
    """Test IntegerAcceptor with single-character advancement."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())

    for char in input_string:
        cursors = list(sm.advance_all(cursors, char))

    print(f"cursors after advancing: {cursors}")
    assert any(cursor.in_accepted_state() for cursor in cursors), f"IntegerAcceptor should accept input '{input_string}'."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"


def test_integer_acceptor_invalid_input():
    """Test IntegerAcceptor with invalid inputs."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    invalid_inputs = ["abc", "12a4", "3.14", "-123", "", "@@@", "123abc456"]

    for input_string in invalid_inputs:
        cursors = list(sm.get_cursors())
        cursors = list(sm.advance_all(cursors, input_string))
        assert not any(cursor.in_accepted_state() for cursor in cursors), f"Input '{input_string}' should not be accepted."


def test_integer_acceptor_empty_input():
    """Test IntegerAcceptor with empty input."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = ""

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))

    assert not any(cursor.in_accepted_state() for cursor in cursors), "Empty input should not be accepted."


def test_integer_acceptor_partial_input():
    """Test IntegerAcceptor with input containing invalid characters."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "12a34"

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))

    assert not any(cursor.in_accepted_state() for cursor in cursors), "Input with invalid characters should not be accepted."


def test_integer_acceptor_in_state_machine_sequence():
    """Test IntegerAcceptor within a StateMachine sequence along with other acceptors."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Number: "), 1)],
            1: [(integer_acceptor, 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    input_string = "Number: 42"

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))

    assert any(cursor.in_accepted_state() for cursor in cursors), "Combined text and integer input should be accepted."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            expected_value = "Number: 42"
            assert value == expected_value, f"Expected '{expected_value}', got '{value}'"


def test_integer_acceptor_char_by_char_in_state_machine():
    """Test IntegerAcceptor within a StateMachine sequence, advancing one character at a time."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Value: "), 1)],
            1: [(integer_acceptor, 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    input_string = "Value: 9876"
    cursors = list(sm.get_cursors())
    for char in input_string:
        cursors = list(sm.advance_all(cursors, char))

    assert any(cursor.in_accepted_state() for cursor in cursors), "Combined text and integer input should be accepted when advancing char by char."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            expected_value = "Value: 9876"
            assert value == expected_value, f"Expected '{expected_value}', got '{value}'"


def test_integer_acceptor_zero():
    """Test IntegerAcceptor with zero."""
    integer_acceptor = IntegerAcceptor()
    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "0"

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))

    assert any(cursor.in_accepted_state() for cursor in cursors), "Zero should be accepted."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            assert value == 0, f"Expected 0, got {value}"


def test_integer_acceptor_large_number():
    """Test IntegerAcceptor with a large number."""
    integer_acceptor = IntegerAcceptor()
    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "12345678901234567890"

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))

    assert any(cursor.in_accepted_state() for cursor in cursors), "Large numbers should be accepted."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            # The parsed value may be a string if it's too large for an int
            if isinstance(value, int):
                assert value == int(input_string), f"Expected {input_string}, got {value}"
            else:
                assert value == input_string, f"Expected '{input_string}', got '{value}'"

@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("0000", 0),
        ("007", 7),
        ("000000", 0),
        ("000001", 1),
    ]
)
def test_integer_acceptor_leading_zeros(input_string, expected_value):
    """Test IntegerAcceptor handling of leading zeros."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(sm.advance_all(cursors, input_string))

    assert any(cursor.in_accepted_state() for cursor in cursors), f"IntegerAcceptor should accept input '{input_string}'."
    for cursor in cursors:
        if cursor.in_accepted_state():
            value = cursor.get_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"


def test_integer_acceptor_cursor_equality():
    """Test the equality of IntegerAcceptor cursors."""
    integer_acceptor = IntegerAcceptor()
    cursor1 = integer_acceptor.Cursor(integer_acceptor, "123")
    cursor2 = integer_acceptor.Cursor(integer_acceptor, "123")
    cursor3 = integer_acceptor.Cursor(integer_acceptor, "124")

    assert cursor1 == cursor2, "Cursors with the same state and value should be equal."
    assert cursor1 != cursor3, "Cursors with different states or values should not be equal."


def test_integer_acceptor_cursor_hash():
    """Test the hash function of IntegerAcceptor cursors."""
    integer_acceptor = IntegerAcceptor()
    cursor_set = set()
    cursor1 = integer_acceptor.Cursor(integer_acceptor, "123")
    cursor2 = integer_acceptor.Cursor(integer_acceptor, "123")
    cursor3 = integer_acceptor.Cursor(integer_acceptor, "124")

    cursor_set.add(cursor1)
    cursor_set.add(cursor2)
    cursor_set.add(cursor3)

    assert len(cursor_set) == 2, "Cursors with the same state and value should have the same hash."


def test_integer_acceptor_cursor_repr():
    """Test the string representation of IntegerAcceptor cursors."""
    integer_acceptor = IntegerAcceptor()
    cursor = integer_acceptor.Cursor(integer_acceptor, "456")
    cursor.current_state = 1
    cursor.value = 456

    repr_str = repr(cursor)
    expected_repr = "IntegerAcceptor.Cursor(text='456', state=1, value=456)"

    assert repr_str == expected_repr, f"Expected repr '{expected_repr}', got '{repr_str}'"

def test_integer_acceptor_cursor_get_value_with_invalid_text():
    """Test get_value method with invalid text in IntegerAcceptor.Cursor."""
    integer_acceptor = IntegerAcceptor()
    cursor = integer_acceptor.Cursor(integer_acceptor, "abc")

    value = cursor.get_value()

    assert value == "abc", f"Expected get_value to return 'abc', got '{value}'"

def test_integer_acceptor_complete_transition_success():
    """Test complete_transition with a valid transition value."""
    integer_acceptor = IntegerAcceptor()
    cursor = integer_acceptor.Cursor(integer_acceptor, "123")

    # Perform complete transition with valid integer
    cursor.complete_transition("4", target_state="1", is_end_state=False)
    result = cursor.complete_transition("5", target_state="$", is_end_state=True)

    assert result is True, "complete_transition should return True on successful transition."
    assert cursor.text == "12345", f"Expected text to be '12345', got '{cursor.text}'."
    assert cursor.current_state == "$", f"Expected state to be '$', got {cursor.current_state}."
    assert cursor.value == 12345, f"Expected value to be 12345, got {cursor.value}."

    assert not cursor.complete_transition(".", target_state=3, is_end_state=False)

def test_integer_acceptor_complete_transition_failure():
    """Test complete_transition with an invalid transition value."""
    integer_acceptor = IntegerAcceptor()
    cursor = integer_acceptor.Cursor(integer_acceptor, "123")

    # Perform complete transition with invalid integer
    result = cursor.complete_transition("1ab", target_state=2, is_end_state=True)

    assert not result, "complete_transition should return False for invalid input."
