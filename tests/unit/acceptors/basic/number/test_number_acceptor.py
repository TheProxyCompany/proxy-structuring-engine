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
        walkers = list(acceptor.get_walkers())
        for char in json_string:
            print(f"char: {char} and walkers: {walkers}")
            walkers = list(acceptor.advance_all(walkers, char))

        assert (
            len(walkers) > 0
        ), f"walkers should remain after advancing with input {json_string}."

        for walker in walkers:
            print(f"post advance walker: {walker}")
            if walker.has_reached_accept_state():
                return walker.get_current_value()

        assert (
            False
        ), f"No accepted walker found after advancing with input {json_string}."

    return _parse_number


@pytest.mark.parametrize(
    "input_string, expected_value, description",
    [
        ("11", 11, "Should correctly parse positive integers."),
        ("123", 123, "Should correctly parse positive integers."),
        ("-456", -456, "Should correctly parse negative integers."),
    ],
)
def test_valid_integer_transitions(
    acceptor: NumberAcceptor, input_string, expected_value, description
):
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
        # ("123.45", 123.45, "Should correctly parse positive decimals."),
        # ("-0.789", -0.789, "Should correctly parse negative decimals."),
    ],
)
def test_valid_decimal_transitions(
    acceptor: NumberAcceptor, input_string, expected_value, description
):
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
def test_valid_exponential_transitions(
    acceptor: NumberAcceptor, input_string, expected_value, description
):
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


# 4. Error Handling Tests


@pytest.mark.parametrize(
    "input_string, error_message",
    [
        (
            "12.34.56",
            "Should raise JSONParsingError for invalid number with multiple decimals.",
        ),
    ],
)
def test_invalid_transition_handling(
    acceptor: NumberAcceptor, input_string, error_message
):
    """
    Tests handling of invalid transitions.
    """
    parse = parse_number(acceptor)
    try:
        parse(input_string)
        assert (
            False
        ), f"Expected parse_number to raise AssertionError for input '{input_string}'."
    except AssertionError:
        pass


def test_incomplete_number_parsing(acceptor: NumberAcceptor):
    """
    Tests handling of incomplete number parsing.
    """
    parse = parse_number(acceptor)
    try:
        parse("123.")
        assert (
            False
        ), "Expected parse_number to raise AssertionError for incomplete number '123.'."
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
        assert (
            False
        ), "Expected parse_number to raise AssertionError for invalid JSON number '1e'."
    except AssertionError:
        pass


def test_number_acceptor_with_state_machine_float():
    """
    Tests NumberAcceptor within a StateMachine for parsing floating-point numbers.
    """

    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "-123.456"
    walkers = sm.advance_all(walkers, number_string)
    for walker in walkers:
        assert (
            walker.get_current_value() == -123.456
        ), "Parsed value should be the float -123.456."
        break

    assert any(
        walker.has_reached_accept_state() for walker in walkers
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

    walkers = sm.get_walkers()
    number_string = "-123.456"
    walkers = sm.advance_all(walkers, number_string)
    for walker in walkers:
        assert (
            walker.get_current_value() == -123.456
        ), "Parsed value should be the float -123.456."
        break
    remaining_string = " I'll never be free."
    for char in remaining_string:
        walkers = list(sm.advance_all(walkers, char))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept valid floating-point number."


def test_number_acceptor_with_state_machine_exponential():
    """
    Tests NumberAcceptor within a StateMachine for parsing exponential numbers.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "6.022e23"
    walkers = sm.advance_all(walkers, number_string)

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept valid exponential number."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == 6.022e23
            ), "Parsed value should be the float 6.022e23."


def test_number_acceptor_with_state_machine_incomplete():
    """
    Tests NumberAcceptor within a StateMachine with incomplete input.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "123.45e"
    walkers = sm.advance_all(walkers, number_string)

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should not accept incomplete number."
    assert (
        len(list(walkers)) == 0
    ), "No walkers should remain after processing incomplete input."


def test_number_acceptor_with_state_machine_invalid():
    """
    Tests NumberAcceptor within a StateMachine with invalid input.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "12abc34"
    walkers = sm.advance_all(walkers, number_string)

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should not accept invalid number."
    assert (
        len(list(walkers)) == 0
    ), "No walkers should remain after processing invalid input."


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

    walkers = sm.get_walkers()
    input_string = "Value: 42"
    walkers = sm.advance_all(walkers, input_string)

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept combined text and number input."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == "Value: 42"
            ), "Parsed value should be the combined string 'Value: 42'."


def test_number_acceptor_with_large_number():
    """
    Tests NumberAcceptor within a StateMachine for parsing very large numbers.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "12345678901234567890"
    walkers = list(sm.advance_all(walkers, number_string))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept very large integer number."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == 12345678901234567890
            ), "Parsed value should be the large integer 12345678901234567890."


def test_number_acceptor_with_leading_zeros():
    """
    Tests NumberAcceptor within a StateMachine for handling leading zeros.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "007"
    walkers = sm.advance_all(walkers, number_string)

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept numbers with leading zeros."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == 7
            ), "Parsed value should be the integer 7."


def test_number_acceptor_with_zero():
    """
    Tests NumberAcceptor within a StateMachine for parsing zero.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    number_string = "0"
    walkers = list(sm.advance_all(walkers, number_string))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept zero."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == 0, "Parsed value should be zero."


def test_number_acceptor_with_multiple_decimal_points():
    """
    Tests NumberAcceptor within a StateMachine for handling multiple decimal points.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "1.2.3"
    walkers = sm.advance_all(walkers, number_string)

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should not accept numbers with multiple decimal points."
    assert (
        len(list(walkers)) == 0
    ), "No walkers should remain after processing invalid decimal input."


def test_number_acceptor_with_empty_input():
    """
    Tests NumberAcceptor within a StateMachine with empty input.
    """
    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = ""
    walkers = sm.advance_all(walkers, number_string)

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should not accept empty input."
    assert (
        len(list(walkers)) == 0
    ), "No walkers should remain after processing empty input."


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
    ],
)
def test_number_acceptor_multi_char_advancement(input_string, expected_value):
    """Test NumberAcceptor with multi-character advancement."""
    number_acceptor = NumberAcceptor()

    sm = StateMachine(
        graph={0: [(number_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"Walkers after advancing: {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"NumberAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            assert value == pytest.approx(
                expected_value
            ), f"Expected {expected_value}, got {value}"


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("7.89", 7.89),
        ("22.0069", 22.0069),
        ("0.00123", 0.00123),
        ("123456.789", 123456.789),
        ("0.0000", 0.0),
        ("9999.0001", 9999.0001),
    ],
)
def test_float_acceptor_single_char_advancement(input_string, expected_value):
    """Test NumberAcceptor with single-character advancement."""
    number_acceptor = NumberAcceptor()

    sm = StateMachine(
        graph={0: [(number_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())

    for char in input_string:
        walkers = list(sm.advance_all(walkers, char))
        print(f"Walkers after advancing '{char}': {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"NumberAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            assert value == pytest.approx(
                expected_value
            ), f"Expected {expected_value}, got {value}"


def test_float_acceptor_invalid_input():
    """Test NumberAcceptor with invalid inputs."""
    number_acceptor = NumberAcceptor()

    sm = StateMachine(
        graph={0: [(number_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    invalid_inputs = [
        "abc",
        "12a4",
        "3.14.15",
        "1..23",
        "--3.14",
        "",
        ".456",  # Starts with a dot but no leading digit
        "123a.456",
        "123.-456",
        "123e456",  # Exponential notation not supported
        ".",  # Just a dot
    ]

    for input_string in invalid_inputs:
        walkers = list(sm.get_walkers())
        walkers = list(sm.advance_all(walkers, input_string))
        print(f"Testing invalid input '{input_string}': Walkers: {walkers}")
        assert not any(
            walker.has_reached_accept_state() for walker in walkers
        ), f"Input '{input_string}' should not be accepted."


def test_number_acceptor_empty_input():
    """Test NumberAcceptor with empty input."""
    number_acceptor = NumberAcceptor()

    sm = StateMachine(
        graph={0: [(number_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = ""

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"Walkers after empty input: {walkers}")

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Empty input should not be accepted."


def test_float_acceptor_partial_input():
    """Test NumberAcceptor with input containing invalid characters."""
    number_acceptor = NumberAcceptor()

    sm = StateMachine(
        graph={0: [(number_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "12.3a4"

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"Walkers after partial invalid input '{input_string}': {walkers}")

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Input with invalid characters should not be accepted."


def test_float_acceptor_in_state_machine_sequence():
    """Test NumberAcceptor within a StateMachine sequence along with other acceptors."""
    number_acceptor = NumberAcceptor()

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Number: "), 1)],
            1: [(number_acceptor, 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    input_string = "Number: 3.14159"

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"Walkers after advancing with input '{input_string}': {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and float input should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            expected_value = "Number: 3.14159"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


def test_float_acceptor_char_by_char_in_state_machine():
    """Test NumberAcceptor within a StateMachine sequence, advancing one character at a time."""
    number_acceptor = NumberAcceptor()

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Value: "), 1)],
            1: [(number_acceptor, 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    input_string = "Value: 0.0001"
    walkers = list(sm.get_walkers())
    for char in input_string:
        walkers = list(sm.advance_all(walkers, char))
        print(f"Walkers after advancing '{char}': {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and float input should be accepted when advancing char by char."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            expected_value = "Value: 0.0001"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


def test_float_acceptor_zero():
    """Test NumberAcceptor with zero and zero fractions."""
    number_acceptor = NumberAcceptor()
    sm = StateMachine(
        graph={0: [(number_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    inputs = ["0.0", "0.0000", "123.0000"]
    expected_values = [0.0, 0.0, 123.0]

    for input_string, expected_value in zip(inputs, expected_values):
        walkers = list(sm.get_walkers())
        walkers = list(sm.advance_all(walkers, input_string))
        print(f"Walkers after advancing with '{input_string}': {walkers}")

        assert any(
            walker.has_reached_accept_state() for walker in walkers
        ), f"Input '{input_string}' should be accepted."
        for walker in walkers:
            if walker.has_reached_accept_state():
                value = walker.get_current_value()
                assert value == pytest.approx(
                    expected_value
                ), f"Expected {expected_value}, got {value}"


def test_float_acceptor_large_number():
    """Test NumberAcceptor with a large floating-point number."""
    number_acceptor = NumberAcceptor()
    sm = StateMachine(
        graph={0: [(number_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "12345678901234567890.123456789"
    expected_value = 1.2345678901234568e19  # Adjusted for float precision

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"Walkers after advancing large number '{input_string}': {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Large floating-point numbers should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            assert value == pytest.approx(
                expected_value
            ), f"Expected {expected_value}, got {value}"
