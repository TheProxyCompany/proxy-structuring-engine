import pytest
from typing import Any
from pse.acceptors.basic.number.number_acceptor import NumberAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor

@pytest.fixture
def acceptor():
    return NumberAcceptor()

def parse_number(acceptor: NumberAcceptor):
    def _parse_number(json_string: str) -> Any:
        """
        Parses a JSON number string using the NumberAcceptor.

        Args:
            json_string (str): The JSON number string to parse.

        Returns:
            Any: The parsed number as an int or float.

        Raises:
            AssertionError: If the JSON number is invalid.
        """
        cursors = list(acceptor.get_cursors())
        for char in json_string:
            print(f"char: {char} and cursors: {cursors}")
            cursors = list(acceptor.advance_all(cursors, char))

        assert len(cursors) > 0, f"cursors should remain after advancing with input {json_string}."

        for cursor in cursors:
            print(f"post advance cursor: {cursor}")
            if cursor.in_accepted_state():
                return cursor.get_value()

        assert False, f"No accepted cursor found after advancing with input {json_string}."
    return _parse_number

@pytest.mark.parametrize(
    "input_string, expected_value, description",
    [
        ("11", 11, "Should correctly parse positive integers."),
        ("123", 123, "Should correctly parse positive integers."),
        ("-456", -456, "Should correctly parse negative integers."),
    ],
)
def test_valid_integer_transitions(acceptor: NumberAcceptor, input_string, expected_value, description):
    """
    Tests valid integer transitions.
    """
    parse = parse_number(acceptor)
    value = parse(input_string)
    assert value == expected_value, description

@pytest.mark.parametrize(
    "input_string, expected_value, description",
    [
        ("1.1", 1.1, "Should correctly parse positive decimals."),
        ("123.45", 123.45, "Should correctly parse positive decimals."),
        ("-0.789", -0.789, "Should correctly parse negative decimals."),
    ],
)
def test_valid_decimal_transitions(acceptor: NumberAcceptor, input_string, expected_value, description):
    """
    Tests valid decimal transitions.
    """
    parse = parse_number(acceptor)
    value = parse(input_string)
    assert value == expected_value, description

@pytest.mark.parametrize(
    "input_string, expected_value, description",
    [
        ("1e10", 1e10, "Should correctly parse positive exponentials."),
        ("-2.5E-3", -2.5e-3, "Should correctly parse negative exponentials."),
    ],
)
def test_valid_exponential_transitions(acceptor: NumberAcceptor, input_string, expected_value, description):
    """
    Tests valid exponential transitions.
    """
    parse = parse_number(acceptor)
    value = parse(input_string)
    assert value == expected_value, description

def test_optional_sign_handling(acceptor: NumberAcceptor):
    """
    Tests handling of optional negative sign.
    """
    parse = parse_number(acceptor)
    value = parse("-456")
    assert value == -456, "Should correctly handle optional negative sign."

# 3. Cursor Behavior Tests

def test_cursor_initialization(acceptor: NumberAcceptor):
    """
    Tests initialization of the Cursor.
    """
    cursor = acceptor.Cursor(acceptor)
    assert cursor.text == "", "Cursor text should be initialized to an empty string."
    assert cursor.value is None, "Cursor value should be initialized to None."

def test_cursor_get_value(acceptor: NumberAcceptor):
    """
    Tests the get_value method of the Cursor.
    """
    cursor = acceptor.Cursor(acceptor)
    cursor.text = "123"
    cursor.value = 123
    assert cursor.get_value() == 123, "get_value should return the parsed integer."

    cursor.text = "45"
    cursor.value = None
    assert cursor.get_value() == "45", "get_value should return the accumulated string when value is None."

# 4. Error Handling Tests

@pytest.mark.parametrize(
    "input_string, error_message",
    [
        ("12.34.56", "Should raise JSONParsingError for invalid number with multiple decimals."),
    ],
)
def test_invalid_transition_handling(acceptor: NumberAcceptor, input_string, error_message):
    """
    Tests handling of invalid transitions.
    """
    parse = parse_number(acceptor)
    try:
        parse(input_string)
        assert False, f"Expected parse_number to raise AssertionError for input '{input_string}'."
    except AssertionError:
        pass

def test_incomplete_number_parsing(acceptor: NumberAcceptor):
    """
    Tests handling of incomplete number parsing.
    """
    parse = parse_number(acceptor)
    try:
        parse("123.")
        assert False, "Expected parse_number to raise AssertionError for incomplete number '123.'."
    except AssertionError:
        pass

def test_zero_handling(acceptor: NumberAcceptor):
    """
    Tests handling of zero and multiple leading zeros.
    """
    parse = parse_number(acceptor)
    value = parse("0")
    assert value == 0, "Should correctly parse zero."

def test_large_numbers(acceptor: NumberAcceptor):
    """
    Tests handling of large numbers.
    """
    parse = parse_number(acceptor)
    value = parse("1234567890123456789")
    assert value == 1234567890123456789, "Should correctly parse very large integers."
    value = parse("1e308")
    assert value == 1e308, "Should correctly parse very large exponentials."

def test_multiple_decimal_points(acceptor: NumberAcceptor):
    """
    Tests handling of multiple decimal points in number.
    """
    parse = parse_number(acceptor)
    try:
        parse("1.2.3")
        assert False, "Expected parse_number to raise AssertionError for multiple decimal points '1.2.3'."
    except AssertionError:
        pass

def test_invalid_characters(acceptor: NumberAcceptor):
    """
    Tests handling of invalid characters in number.
    """
    parse = parse_number(acceptor)
    try:
        parse("12a34")
        assert False, "Expected parse_number to raise AssertionError for invalid characters '12a34'."
    except AssertionError:
        pass

def test_invalid_json_number(acceptor: NumberAcceptor):
    """
    Test that AssertionError is raised for invalid JSON numbers at end state.
    """
    parse = parse_number(acceptor)
    try:
        parse("1e")
        assert False, "Expected parse_number to raise AssertionError for invalid JSON number '1e'."
    except AssertionError:
        pass

def test_number_acceptor_with_state_machine_float():
    """
    Tests NumberAcceptor within a StateMachine for parsing floating-point numbers.
    """

    sm = StateMachine(
        graph={
            0: [(NumberAcceptor(), 1)]
        },
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = "-123.456"
    cursors = sm.advance_all(cursors, number_string)
    for cursor in cursors:
        assert cursor.get_value() == -123.456, "Parsed value should be the float -123.456."
        break

    assert any(
        cursor.in_accepted_state() for cursor in cursors
    ), "StateMachine should accept valid floating-point number."

def test_number_acceptor_with_state_machine_float_complex():
    """
    Tests NumberAcceptor within a StateMachine for parsing floating-point numbers.
    """
    sm = StateMachine(
        graph={
            0: [(NumberAcceptor(), 1)],
            1: [(WhitespaceAcceptor(), 2)],
            2: [(TextAcceptor("I'll never be free."), "$")],
        },
        initial_state=0,
        end_states=["$"],
    )

    cursors = sm.get_cursors()
    number_string = "-123.456"
    cursors = sm.advance_all(cursors, number_string)
    for cursor in cursors:
        assert cursor.get_value() == -123.456, "Parsed value should be the float -123.456."
        break
    remaining_string = " I'll never be free."
    for char in remaining_string:
        cursors = list(sm.advance_all(cursors, char))

    assert any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should accept valid floating-point number."

def test_number_acceptor_with_state_machine_exponential():
    """
    Tests NumberAcceptor within a StateMachine for parsing exponential numbers.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = "6.022e23"
    cursors = sm.advance_all(cursors, number_string)

    assert any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should accept valid exponential number."
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == 6.022e23, "Parsed value should be the float 6.022e23."

def test_number_acceptor_with_state_machine_incomplete():
    """
    Tests NumberAcceptor within a StateMachine with incomplete input.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = "123.45e"
    cursors = sm.advance_all(cursors, number_string)

    assert not any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should not accept incomplete number."
    assert len(list(cursors)) == 0, "No cursors should remain after processing incomplete input."

def test_number_acceptor_with_state_machine_invalid():
    """
    Tests NumberAcceptor within a StateMachine with invalid input.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = "12abc34"
    cursors = sm.advance_all(cursors, number_string)

    assert not any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should not accept invalid number."
    assert len(list(cursors)) == 0, "No cursors should remain after processing invalid input."

def test_number_acceptor_in_state_machine_sequence():
    """
    Tests NumberAcceptor within a StateMachine sequence along with other acceptors.
    """
    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Value: "), 1)],
            1: [(NumberAcceptor(), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    cursors = sm.get_cursors()
    input_string = "Value: 42"
    cursors = sm.advance_all(cursors, input_string)

    assert any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should accept combined text and number input."
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "Value: 42", "Parsed value should be the combined string 'Value: 42'."

def test_number_acceptor_with_large_number():
    """
    Tests NumberAcceptor within a StateMachine for parsing very large numbers.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = "12345678901234567890"
    cursors = list(sm.advance_all(cursors, number_string))

    assert any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should accept very large integer number."
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == 12345678901234567890, "Parsed value should be the large integer 12345678901234567890."

def test_number_acceptor_with_leading_zeros():
    """
    Tests NumberAcceptor within a StateMachine for handling leading zeros.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = "007"
    cursors = sm.advance_all(cursors, number_string)

    assert any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should accept numbers with leading zeros."
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == 7, "Parsed value should be the integer 7."

def test_number_acceptor_with_zero():
    """
    Tests NumberAcceptor within a StateMachine for parsing zero.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    number_string = "0"
    cursors = list(sm.advance_all(cursors, number_string))

    assert any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should accept zero."
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == 0, "Parsed value should be zero."

def test_number_acceptor_with_multiple_decimal_points():
    """
    Tests NumberAcceptor within a StateMachine for handling multiple decimal points.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = "1.2.3"
    cursors = sm.advance_all(cursors, number_string)

    assert not any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should not accept numbers with multiple decimal points."
    assert len(list(cursors)) == 0, "No cursors should remain after processing invalid decimal input."

def test_number_acceptor_with_empty_input():
    """
    Tests NumberAcceptor within a StateMachine with empty input.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = sm.get_cursors()
    number_string = ""
    cursors = sm.advance_all(cursors, number_string)

    assert not any(cursor.in_accepted_state() for cursor in cursors), "StateMachine should not accept empty input."
    assert len(list(cursors)) == 0, "No cursors should remain after processing empty input."
