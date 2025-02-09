import pytest
from pse_core._core import Stepper
from pse_core.state_machine import StateMachine

from pse.types.base.phrase import PhraseStateMachine
from pse.types.number import NumberStateMachine


@pytest.fixture
def state_machine() -> NumberStateMachine:
    """
    Fixture to provide a NumberAcceptor instance.

    Returns:
        NumberAcceptor: An instance of NumberAcceptor.
    """
    return NumberStateMachine()


# Test Cases for Valid Decimal Numbers
@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("1.1", 1.1),
        ("123.45", 123.45),
        ("-0.789", -0.789),
        ("0.0", 0.0),
        ("-0.0", 0.0),
        ("1.0", 1.0),
        ("123.0000", 123.0),
        ("0.0001", 0.0001),
        ("98765.4321", 98765.4321),
    ],
)
def test_valid_decimal_numbers(input_string: str, expected_value: float) -> None:
    """
    Test parsing of valid decimal numbers.

    Args:
        input_string (str): The decimal number string to parse.
        expected_value (float): The expected parsed value.
        description (str): Description of the test case.
    """
    sm = NumberStateMachine()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_string)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert expected_value == pytest.approx(stepper.get_current_value())


# Test Cases for Valid Exponential Numbers
@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("1e10", 1e10),
        ("-2.5e-3", -2.5e-3),
        ("6.022e23", 6.022e23),
        ("1E+2", 1e2),
        ("-1E-2", -1e-2),
        ("0e0", 0.0),
    ],
)
def test_valid_exponential_numbers(input_string: str, expected_value: float) -> None:
    """
    Test parsing of valid exponential numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
        input_string (str): The exponential number string to parse.
        expected_value (float): The expected parsed value.
        description (str): Description of the test case.
    """
    sm = NumberStateMachine()
    steppers: list[Stepper] = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_string)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert expected_value == pytest.approx(stepper.get_current_value())


# Edge Case Tests
@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("0", 0),
        ("-0", 0),
        ("0.0", 0),
        ("-0.0", 0),
    ],
)
def test_zero_handling(input_string: str, expected_value: float) -> None:
    """
    Test parsing of zero and negative zero.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    sm = NumberStateMachine()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_string)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_string, expected_value",
    [("1e308", 1e308), ("-1e308", -1e308)],
)
def test_large_number_parsing(input_string: str, expected_value: float) -> None:
    """
    Test parsing of very large numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    sm = NumberStateMachine()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_string)
    assert steppers[0].has_reached_accept_state()
    assert expected_value == pytest.approx(steppers[0].get_current_value())


@pytest.mark.parametrize(
    "input_string, expected_value",
    [("007", 7), ("000.123", 0.123), ("123.45000", 123.45)],
)
def test_number_with_leading_zeros(input_string: str, expected_value: float) -> None:
    """
    Test parsing numbers with leading zeros.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """
    sm = NumberStateMachine()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_string)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert expected_value == pytest.approx(stepper.get_current_value())


# Error Handling Tests
@pytest.mark.parametrize(
    "input_string",
    [
        "12.34.56",
        "1e+e",
        "1e",
        "1e1.5",
        ".456",
        "abc",
        "--123",
        "123.",
        "12a34",
        "1.2.3",
        "1e10e5",
    ],
)
def test_invalid_number_parsing(input_string: str) -> None:
    """
    Test parsing of invalid numbers.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
        input_string (str): The invalid number string to parse.
    """
    sm = NumberStateMachine()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_string)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


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

    steppers = sm.get_steppers()
    number_string = "-123.456"
    steppers = sm.advance_all_basic(steppers, number_string)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StateMachine should accept valid floating-point number."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == -123.456, (
                "Parsed value should be the float -123.456."
            )


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

    steppers = sm.get_steppers()
    number_string = "6.022e23"
    steppers = sm.advance_all_basic(steppers, number_string)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StateMachine should accept valid exponential number."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == 6.022e23, (
                "Parsed value should be the float 6.022e23."
            )


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
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Combined text and number input should be accepted."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            expected_value = "Value: 42.0"
            assert value == expected_value, (
                f"Expected '{expected_value}', got '{value}'"
            )


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

    steppers = sm.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"NumberAcceptor should accept input '{input_string}'."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value == pytest.approx(expected_value), (
                f"Expected {expected_value}, got {value}"
            )


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

    steppers = sm.get_steppers()

    for char in input_string:
        steppers = sm.advance_all_basic(steppers, char)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"NumberAcceptor should accept input '{input_string}'."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value == pytest.approx(expected_value), (
                f"Expected {expected_value}, got {value}"
            )


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
        steppers = sm.get_steppers()
        steppers = state_machine.advance_all_basic(steppers, input_string)
        assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
            f"Input '{input_string}' should not be accepted."
        )


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

    steppers = sm.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, input_string)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Empty input should not be accepted."
    )


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

    steppers = sm.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, input_string)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Input with invalid characters should not be accepted."
    )


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

    steppers = sm.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Large floating-point numbers should be accepted."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value == pytest.approx(expected_value), (
                f"Expected {expected_value}, got {value}"
            )


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

    steppers = sm.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, "Value: ")
    assert len(steppers) == 2
    steppers = state_machine.advance_all_basic(steppers, str(value))

    assert len(steppers) == 4, "Should have four steppers."
    if followup_value is not None:
        steppers = state_machine.advance_all_basic(steppers, str(followup_value))

    steppers = state_machine.advance_all_basic(steppers, "!")
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert (
                stepper.get_current_value()
                == f"Value: {str(value) + str(followup_value or '')}!"
            )
