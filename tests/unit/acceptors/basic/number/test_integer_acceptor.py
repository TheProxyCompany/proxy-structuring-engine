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
    ],
)
def test_integer_acceptor_multi_char_advancement(input_string, expected_value):
    """Test IntegerAcceptor with multi-character advancement."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    print(f"walkers: {walkers}")
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"walkers: {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"IntegerAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("7", 7),
        ("89", 89),
        ("123456", 123456),
        # ("000789", 789),  # Leading zeros should be handled
    ],
)
def test_integer_acceptor(input_string, expected_value):
    """Test IntegerAcceptor with single-character advancement."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    for char in input_string:
        print(f"Advancing with {char}, walkers: {walkers}")
        walkers = list(sm.advance_all(walkers, char))

    print(f"walkers after advancing: {walkers}")
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"IntegerAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
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
        walkers = list(sm.get_walkers())
        walkers = list(sm.advance_all(walkers, input_string))
        assert not any(
            walker.has_reached_accept_state() for walker in walkers
        ), f"Input '{input_string}' should not be accepted."


def test_integer_acceptor_empty_input():
    """Test IntegerAcceptor with empty input."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = ""

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Empty input should not be accepted."


def test_integer_acceptor_partial_input():
    """Test IntegerAcceptor with input containing invalid characters."""
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "12a34"

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Input with invalid characters should not be accepted."


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

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and integer input should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            expected_value = "Number: 42"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


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
    walkers = list(sm.get_walkers())
    for char in input_string:
        walkers = list(sm.advance_all(walkers, char))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and integer input should be accepted when advancing char by char."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            expected_value = "Value: 9876"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


def test_integer_acceptor_zero():
    """Test IntegerAcceptor with zero."""
    integer_acceptor = IntegerAcceptor()
    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    input_string = "0"

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Zero should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
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

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Large numbers should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            # The parsed value may be a string if it's too large for an int
            if isinstance(value, int):
                assert value == int(
                    input_string
                ), f"Expected {input_string}, got {value}"
            else:
                assert (
                    value == input_string
                ), f"Expected '{input_string}', got '{value}'"


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
    integer_acceptor = IntegerAcceptor()

    sm = StateMachine(
        graph={0: [(integer_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"IntegerAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            assert value == expected_value, f"Expected {expected_value}, got {value}"


def test_integer_acceptor_walker_equality():
    """Test the equality of IntegerAcceptor walkers."""
    integer_acceptor = IntegerAcceptor()
    walker1 = integer_acceptor.walker_class(integer_acceptor, "123")
    walker2 = integer_acceptor.walker_class(integer_acceptor, "123")
    walker3 = integer_acceptor.walker_class(integer_acceptor, "124")

    assert walker1 == walker2, "Walkers with the same state and value should be equal."
    assert (
        walker1 != walker3
    ), "Walkers with different states or values should not be equal."


def test_integer_acceptor_walker_hash():
    """Test the hash function of IntegerAcceptor walkers."""
    integer_acceptor = IntegerAcceptor()
    walker_set = set()
    walker1 = integer_acceptor.walker_class(integer_acceptor, "123")
    walker2 = integer_acceptor.walker_class(integer_acceptor, "123")
    walker3 = integer_acceptor.walker_class(integer_acceptor, "124")

    walker_set.add(walker1)
    walker_set.add(walker2)
    walker_set.add(walker3)

    assert (
        len(walker_set) == 2
    ), "Walkers with the same state and value should have the same hash."


def test_integer_acceptor_walker_get_value_with_invalid_text():
    """Test get_value method with invalid text in IntegerAcceptor.Walker."""
    integer_acceptor = IntegerAcceptor()
    walker = integer_acceptor.walker_class(integer_acceptor, "abc")

    value = walker.accumulated_value()

    assert value == "abc", f"Expected get_value to return 'abc', got '{value}'"


def test_integer_acceptor_complete_transition_success():
    """Test complete_transition with a valid transition value."""
    integer_acceptor = IntegerAcceptor()
    walker = integer_acceptor.walker_class(integer_acceptor, "123")

    # Perform complete transition with valid integer
    walker.should_complete_transition("4", target_state="1", is_end_state=False)
    result = walker.should_complete_transition("5", target_state="$", is_end_state=True)

    assert (
        result is True
    ), "complete_transition should return True on successful transition."
    assert walker.text == "12345", f"Expected text to be '12345', got '{walker.text}'."
    assert (
        walker.current_state == "$"
    ), f"Expected state to be '$', got {walker.current_state}."
    assert walker.value == 12345, f"Expected value to be 12345, got {walker.value}."

    assert not walker.should_complete_transition(
        ".", target_state=3, is_end_state=False
    )


def test_integer_acceptor_complete_transition_failure():
    """Test complete_transition with an invalid transition value."""
    integer_acceptor = IntegerAcceptor()
    walker = integer_acceptor.walker_class(integer_acceptor, "123")

    # Perform complete transition with invalid integer
    result = walker.should_complete_transition("1ab", target_state=2, is_end_state=True)

    assert not result, "complete_transition should return False for invalid input."

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
def test_float_acceptor_multi_char_advancement(input_string, expected_value):
    """Test FloatAcceptor with multi-character advancement."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"Walkers after advancing: {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"FloatAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
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
    """Test FloatAcceptor with single-character advancement."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())

    for char in input_string:
        walkers = list(sm.advance_all(walkers, char))
        print(f"Walkers after advancing '{char}': {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"FloatAcceptor should accept input '{input_string}'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            assert value == pytest.approx(
                expected_value
            ), f"Expected {expected_value}, got {value}"


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
        ".",  # Just a dot
    ]

    for input_string in invalid_inputs:
        walkers = list(sm.get_walkers())
        walkers = list(sm.advance_all(walkers, input_string))
        print(f"Testing invalid input '{input_string}': Walkers: {walkers}")
        assert not any(
            walker.has_reached_accept_state() for walker in walkers
        ), f"Input '{input_string}' should not be accepted."


def test_float_acceptor_empty_input():
    """Test FloatAcceptor with empty input."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
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
    """Test FloatAcceptor with input containing invalid characters."""
    float_acceptor = FloatAcceptor()

    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
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

    walkers = list(sm.get_walkers())
    walkers = list(sm.advance_all(walkers, input_string))
    print(f"Walkers after advancing with input '{input_string}': {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and float input should be accepted."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            expected_value = "Number: 3.14159"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


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
    walkers = list(sm.get_walkers())
    for char in input_string:
        walkers = list(sm.advance_all(walkers, char))
        print(f"Walkers after advancing '{char}': {walkers}")

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Combined text and float input should be accepted when advancing char by char."
    for walker in walkers:
        if walker.has_reached_accept_state():
            value = walker.accumulated_value()
            expected_value = "Value: 0.0001"
            assert (
                value == expected_value
            ), f"Expected '{expected_value}', got '{value}'"


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
        walkers = list(sm.get_walkers())
        walkers = list(sm.advance_all(walkers, input_string))
        print(f"Walkers after advancing with '{input_string}': {walkers}")

        assert any(
            walker.has_reached_accept_state() for walker in walkers
        ), f"Input '{input_string}' should be accepted."
        for walker in walkers:
            if walker.has_reached_accept_state():
                value = walker.accumulated_value()
                assert value == pytest.approx(
                    expected_value
                ), f"Expected {expected_value}, got {value}"


def test_float_acceptor_large_number():
    """Test FloatAcceptor with a large floating-point number."""
    float_acceptor = FloatAcceptor()
    sm = StateMachine(
        graph={0: [(float_acceptor, 1)]},
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
            value = walker.accumulated_value()
            assert value == pytest.approx(
                expected_value
            ), f"Expected {expected_value}, got {value}"
