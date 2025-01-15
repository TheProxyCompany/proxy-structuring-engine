from typing import Any

import pytest
from pse_core.state_machine import StateMachine

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.types.number import NumberStateMachine


@pytest.fixture
def state_machine() -> NumberStateMachine:
    """
    Fixture to provide a NumberAcceptor instance.

    Returns:
        NumberAcceptor: An instance of NumberAcceptor.
    """
    return NumberStateMachine()


def parse_number(state_machine: NumberStateMachine):
    """
    Helper function to parse a JSON number string using the NumberAcceptor.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.

    Returns:
        Callable[[str], Any]: A function that takes a JSON number string and returns the parsed value.
    """

    def _parse_number(json_string: str) -> Any:
        """
        Parses a JSON number string using the NumberAcceptor.

        Args:
            json_string (str): The JSON number string to parse.

        Returns:
            Any: The parsed number as int or float.

        Raises:
            AssertionError: If no accepted walker is found.
        """
        walkers = list(state_machine.get_walkers())
        for char in json_string:
            walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]

        assert (
            len(walkers) > 0
        ), f"No walkers remain after advancing with input '{json_string}'."

        for walker in walkers:
            if walker.has_reached_accept_state():
                return walker.get_current_value()

        raise AssertionError(
            f"No accepted walker found after advancing with input '{json_string}'."
        )

    return _parse_number


# Test Cases for Valid Decimal Numbers
@pytest.mark.parametrize(
    "input_string, expected_value, description",
    [
        ("1.1", 1.1, "Should correctly parse positive decimals."),
        ("123.45", 123.45, "Should correctly parse positive decimals."),
        ("-0.789", -0.789, "Should correctly parse negative decimals."),
        ("0.0", 0.0, "Should correctly parse zero as decimal."),
        ("-0.0", 0.0, "Should correctly parse negative zero as zero."),
        ("1.0", 1.0, "Should correctly parse decimal with zero fraction."),
        ("123.0000", 123.0, "Should correctly parse decimal with trailing zeros."),
        ("0.0001", 0.0001, "Should correctly parse small decimals."),
        ("98765.4321", 98765.4321, "Should correctly parse large decimals."),
    ],
)
def test_valid_decimal_numbers(
    state_machine: NumberStateMachine,
    input_string: str,
    expected_value: float,
    description: str,
) -> None:
    """
    Test parsing of valid decimal numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
        input_string (str): The decimal number string to parse.
        expected_value (float): The expected parsed value.
        description (str): Description of the test case.
    """
    parse = parse_number(state_machine)
    value = parse(input_string)
    assert value == pytest.approx(expected_value), description


# Test Cases for Valid Exponential Numbers
@pytest.mark.parametrize(
    "input_string, expected_value, description",
    [
        ("1e10", 1e10, "Should correctly parse positive exponentials."),
        ("-2.5e-3", -2.5e-3, "Should correctly parse negative exponentials."),
        ("6.022e23", 6.022e23, "Should correctly parse large exponentials."),
        ("1E+2", 1e2, "Should correctly parse exponentials with '+' sign."),
        ("-1E-2", -1e-2, "Should correctly parse negative exponentials with '-' sign."),
        ("0e0", 0.0, "Should correctly parse zero exponential."),
    ],
)
def test_valid_exponential_numbers(
    state_machine: NumberStateMachine,
    input_string: str,
    expected_value: float,
    description: str,
) -> None:
    """
    Test parsing of valid exponential numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
        input_string (str): The exponential number string to parse.
        expected_value (float): The expected parsed value.
        description (str): Description of the test case.
    """
    parse = parse_number(state_machine)
    value = parse(input_string)
    assert value == pytest.approx(expected_value), description


# Edge Case Tests
def test_zero_handling(state_machine: NumberStateMachine) -> None:
    """
    Test parsing of zero and negative zero.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    parse = parse_number(state_machine)
    value = parse("0")
    assert value == 0, "Should correctly parse zero."
    value = parse("-0")
    assert value == 0, "Should correctly parse negative zero as zero."


def test_large_number_parsing(state_machine: NumberStateMachine) -> None:
    """
    Test parsing of very large numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    parse = parse_number(state_machine)
    value = parse("1e308")
    assert value == 1e308, "Should correctly parse very large exponentials."
    value = parse("-1e308")
    assert value == -1e308, "Should correctly parse very large negative exponentials."


def test_number_with_leading_zeros(state_machine: NumberStateMachine) -> None:
    """
    Test parsing numbers with leading zeros.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    parse = parse_number(state_machine)
    value = parse("007")
    assert value == 7, "Should correctly parse numbers with leading zeros."
    value = parse("000.123")
    assert value == 0.123, "Should correctly parse decimals with leading zeros."


def test_number_with_trailing_zeros(state_machine: NumberStateMachine) -> None:
    """
    Test parsing numbers with trailing zeros.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    parse = parse_number(state_machine)
    value = parse("123.45000")
    assert value == 123.45, "Should correctly parse numbers with trailing zeros."


# Error Handling Tests
@pytest.mark.parametrize(
    "input_string, error_message",
    [
        (
            "12.34.56",
            "Should raise AssertionError for invalid number with multiple decimals.",
        ),
        (
            "1e",
            "Should raise AssertionError for incomplete exponentials.",
        ),
        (
            "1e+e",
            "Should raise AssertionError for invalid exponential format.",
        ),
        (
            "1e1.5",
            "Should raise AssertionError for non-integer exponent.",
        ),
        (
            ".456",
            "Should raise AssertionError for numbers starting with a dot.",
        ),
        (
            "abc",
            "Should raise AssertionError for non-numeric input.",
        ),
        (
            "--123",
            "Should raise AssertionError for multiple negative signs.",
        ),
        (
            "",
            "Should raise AssertionError for empty input.",
        ),
        (
            "12a34",
            "Should raise AssertionError for invalid characters in number.",
        ),
        (
            "123.",
            "Should raise AssertionError for incomplete decimal number.",
        ),
        (
            "1.2.3",
            "Should raise AssertionError for multiple decimal points.",
        ),
        (
            "1e10e5",
            "Should raise AssertionError for multiple exponentials.",
        ),
    ],
)
def test_invalid_number_parsing(
    state_machine: NumberStateMachine, input_string: str, error_message: str
) -> None:
    """
    Test parsing of invalid numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
        input_string (str): The invalid number string to parse.
        error_message (str): Description of the expected error.
    """
    parse = parse_number(state_machine)
    with pytest.raises(AssertionError):
        parse(input_string)


# Tests with StateMachine Integration
def test_number_acceptor_with_state_machine_float(
    state_machine: NumberStateMachine,
) -> None:
    """
    Test NumberAcceptor within a StateMachine for parsing floating-point numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    sm = StateMachine(
        state_graph={0: [(state_machine, 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "-123.456"
    walkers = [walker for _, walker in sm.advance_all(walkers, number_string)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept valid floating-point number."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == -123.456
            ), "Parsed value should be the float -123.456."


def test_number_acceptor_with_state_machine_exponential(
    state_machine: NumberStateMachine,
) -> None:
    """
    Test NumberAcceptor within a StateMachine for parsing exponential numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    sm = StateMachine(
        state_graph={0: [(state_machine, 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    number_string = "6.022e23"
    walkers = [walker for _, walker in sm.advance_all(walkers, number_string)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept valid exponential number."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == 6.022e23
            ), "Parsed value should be the float 6.022e23."


def test_number_acceptor_with_state_machine_sequence(
    state_machine: NumberStateMachine,
) -> None:
    """
    Test NumberAcceptor within a StateMachine sequence along with other acceptors.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("Value: "), 1)],
            1: [(state_machine, 2)],
        },
        start_state=0,
        end_states=[2],
    )

    input_string = "Value: 42.0"
    walkers = sm.get_walkers()
    walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and number input should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            expected_value = "Value: 42.0"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


# Advanced Tests with Character-by-Character Advancement
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
        ("-0.0", 0.0),
        ("-123.456", -123.456),
        ("-98765.4321", -98765.4321),
    ],
)
def test_number_acceptor_multi_char_advancement(
    state_machine: NumberStateMachine, input_string: str, expected_value: float
) -> None:
    """
    Test NumberAcceptor with multi-character advancement.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
        input_string (str): The number string to parse.
        expected_value (float): The expected parsed value.
    """
    sm = StateMachine(
        state_graph={0: [(state_machine, 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

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
        ("-7.89", -7.89),
        ("-0.00123", -0.00123),
    ],
)
def test_number_acceptor_single_char_advancement(
    state_machine: NumberStateMachine, input_string: str, expected_value: float
) -> None:
    """
    Test NumberAcceptor with single-character advancement.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
        input_string (str): The number string to parse.
        expected_value (float): The expected parsed value.
    """
    sm = StateMachine(
        state_graph={0: [(state_machine, 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()

    for char in input_string:
        walkers = [walker for _, walker in sm.advance_all(walkers, char)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"NumberAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            assert value == pytest.approx(
                expected_value
            ), f"Expected {expected_value}, got {value}"


def test_number_acceptor_invalid_input(state_machine: NumberStateMachine) -> None:
    """
    Test NumberAcceptor with invalid inputs.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
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
        "1e1.5",  # Invalid exponent
        ".",  # Just a dot
    ]

    for input_string in invalid_inputs:
        sm = StateMachine(
            state_graph={0: [(state_machine, 1)]},
            start_state=0,
            end_states=[1],
        )
        walkers = sm.get_walkers()
        walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]
        assert not any(
            walker.has_reached_accept_state() for walker in walkers
        ), f"Input '{input_string}' should not be accepted."


def test_number_acceptor_empty_input(state_machine: NumberStateMachine) -> None:
    """
    Test NumberAcceptor with empty input.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    sm = StateMachine(
        state_graph={0: [(state_machine, 1)]},
        start_state=0,
        end_states=[1],
    )
    input_string = ""

    walkers = sm.get_walkers()
    walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Empty input should not be accepted."


def test_number_acceptor_partial_invalid_input(
    state_machine: NumberStateMachine,
) -> None:
    """
    Test NumberAcceptor with input containing invalid characters.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    input_string = "12.3a4"

    sm = StateMachine(
        state_graph={0: [(state_machine, 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Input with invalid characters should not be accepted."


def test_number_acceptor_large_floating_point(
    state_machine: NumberStateMachine,
) -> None:
    """
    Test NumberAcceptor with a large floating-point number.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    input_string = "12345678901234567890.123456789"
    expected_value = 12345678901234567890.123456789

    sm = StateMachine(
        state_graph={0: [(state_machine, 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    walkers = [walker for _, walker in sm.advance_all(walkers, input_string)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Large floating-point numbers should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.get_current_value()
            assert value == pytest.approx(
                expected_value
            ), f"Expected {expected_value}, got {value}"


@pytest.mark.parametrize(
    "value, followup_value",
    [
        (10, None),
        (-10, 1),
        (0, ".0"),
    ],
)
def test_number_acceptor(
    state_machine: NumberStateMachine, value: int, followup_value: int | str | None
) -> None:
    """
    Test NumberAcceptor with an integer.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """

    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("Value: "), 1)],
            1: [(state_machine, 2)],
            2: [(PhraseStateMachine("!"), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    walkers = sm.get_walkers()
    walkers = [walker for _, walker in sm.advance_all(walkers, "Value: ")]
    assert len(walkers) == 2
    walkers = [walker for _, walker in sm.advance_all(walkers, str(value))]

    assert len(walkers) == 4, "Should have four walkers."
    if followup_value is not None:
        walkers = [walker for _, walker in sm.advance_all(walkers, str(followup_value))]

    walkers = [walker for _, walker in sm.advance_all(walkers, "!")]
    assert any(walker.has_reached_accept_state() for walker in walkers)

    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value()
                == f"Value: {str(value) + str(followup_value or "")}!"
            )
