"""Unit tests for AnyCharacterStateMachine and AnyCharacterStepper."""

import pytest

from pse.types.base.character import (
    CharacterStateMachine,
    CharacterStepper,
)


@pytest.fixture
def basic_state_machine():
    """Fixture for a basic AnyCharacterStateMachine with no restrictions."""
    return CharacterStateMachine()


@pytest.fixture
def allowed_charset_state_machine():
    """Fixture for AnyCharacterStateMachine with allowed charset."""
    return CharacterStateMachine(charset=["a", "b", "c"])


@pytest.fixture
def disallowed_charset_state_machine():
    """Fixture for AnyCharacterStateMachine with disallowed charset."""
    return CharacterStateMachine(blacklist_charset=["x", "y", "z"])


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("hello", "hello"),
        ("123", 123),
        ("!@#", "!@#"),
        ("", None),
    ],
)
def test_basic_character_acceptance(basic_state_machine, input_str, expected_value):
    """Test basic character acceptance with no restrictions."""
    stepper = CharacterStepper(basic_state_machine)
    steppers = list(stepper.consume(input_str))

    if not steppers:
        assert expected_value is None
        return

    assert steppers[0].get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("abc", "abc"),
        ("cab", "cab"),
        ("abcd", "abc"),
        ("xyz", None),
    ],
)
def test_allowed_charset(allowed_charset_state_machine, input_str, expected_value):
    """Test character acceptance with allowed charset."""
    stepper = CharacterStepper(allowed_charset_state_machine)
    steppers = stepper.consume(input_str)

    if not steppers:
        assert expected_value is None
        return

    assert steppers[0].get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("abc", "abc"),
        ("def", "def"),
        ("xyz", None),
        ("axy", "a"),
    ],
)
def test_disallowed_charset(
    disallowed_charset_state_machine, input_str, expected_value
):
    """Test character acceptance with disallowed charset."""
    stepper = CharacterStepper(disallowed_charset_state_machine)
    steppers = list(stepper.consume(input_str))

    if not steppers:
        assert expected_value is None
        return

    assert steppers[0].get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("ABC", "ABC"),
        ("abc", None),
        ("AbC", "A"),
    ],
)
def test_case_sensitivity(input_str, expected_value):
    """Test case-sensitive character acceptance."""
    case_sensitive_state_machine = CharacterStateMachine(
        charset=["A", "B", "C"], case_sensitive=True
    )
    stepper = CharacterStepper(case_sensitive_state_machine)
    steppers = list(stepper.consume(input_str))

    if not steppers:
        assert expected_value is None
        return

    assert steppers[0].get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_str, should_accept",
    [
        ("a", False),  # Too short
        ("ab", True),  # Minimum length
        ("abc", True),  # Within limits
        ("abcd", True),  # Maximum length
        ("abcde", False),  # Too long
    ],
)
def test_character_limits(input_str, should_accept):
    """Test character length limits."""
    limited_state_machine = CharacterStateMachine(char_min=2, char_limit=4)
    stepper = CharacterStepper(limited_state_machine)
    steppers = stepper.consume(input_str)

    assert len(steppers) == 1
    if should_accept:
        assert steppers[0].get_current_value() == input_str
        assert steppers[0].has_reached_accept_state()
    else:
        assert steppers[0].remaining_input or not steppers[0].has_reached_accept_state()


def test_stepper_clone(basic_state_machine):
    """Test stepper cloning functionality."""
    stepper = CharacterStepper(basic_state_machine, "test")
    cloned = stepper.clone()

    assert stepper.get_current_value() == cloned.get_current_value()
    assert stepper is not cloned

    # Verify independent advancement
    stepper = stepper.consume("a")[0]
    assert stepper.get_current_value() == "testa"
    assert cloned.get_current_value() == "test"


def test_stepper_is_within_value(basic_state_machine):
    """Test is_within_value method."""
    stepper = CharacterStepper(basic_state_machine)
    assert stepper.is_within_value()

    list(stepper.consume("test"))
    assert stepper.is_within_value()


def test_should_start_transition(basic_state_machine):
    """Test should_start_step method."""
    stepper = CharacterStepper(basic_state_machine)

    assert stepper.should_start_step("a")
    assert stepper.should_start_step("1")
    assert stepper.should_start_step("!")
    assert not stepper.should_start_step("")


def test_optional_state_machine():
    """Test optional AnyCharacterStateMachine."""
    sm = CharacterStateMachine(is_optional=True)
    stepper = CharacterStepper(sm)

    # Should accept empty input
    assert stepper.get_current_value() is None

    # Should also accept valid input
    steppers = list(stepper.consume("a"))
    assert len(steppers) == 1
    assert steppers[0].get_current_value() == "a"
