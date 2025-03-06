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
    return CharacterStateMachine(whitelist_charset=["a", "b", "c"])


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
        whitelist_charset=["A", "B", "C"], case_sensitive=True
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
    assert not stepper.is_within_value()

    stepper = stepper.consume("test")[0]
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


def test_case_insensitive_character():
    """Test case-insensitive character acceptance."""
    ci_state_machine = CharacterStateMachine(
        whitelist_charset=["A", "B", "C"], case_sensitive=False
    )
    stepper = CharacterStepper(ci_state_machine)

    # Should accept both uppercase and lowercase with case-insensitive matching
    steppers = list(stepper.consume("a"))
    assert len(steppers) == 1
    # Value in lowercase per implementation (token is lowercase in consume())
    assert steppers[0].get_current_value() == "a"

    steppers = list(stepper.consume("A"))
    assert len(steppers) == 1
    # Also lowercase because of case_sensitive=False
    assert steppers[0].get_current_value() == "a"

    # Blacklist should also be case-insensitive
    bl_state_machine = CharacterStateMachine(
        blacklist_charset=["X", "Y", "Z"], case_sensitive=False
    )
    stepper = CharacterStepper(bl_state_machine)

    steppers = list(stepper.consume("x"))
    assert len(steppers) == 0  # Should not accept lowercase 'x'

    steppers = list(stepper.consume("X"))
    assert len(steppers) == 0  # Should not accept uppercase 'X'

    # Test graylist charset
    graylist_state_machine = CharacterStateMachine(
        graylist_charset=["d", "e", "f"], case_sensitive=False
    )
    stepper = graylist_state_machine.get_new_stepper("")
    steppers = list(stepper.consume("abD"))
    assert len(steppers) == 1
    assert steppers[0].get_current_value() == "ab"
    assert steppers[0].remaining_input == "d"

    steppers = list(stepper.consume("a"))
    assert len(steppers) == 1
    assert steppers[0].get_current_value() == "a"

    steppers = list(stepper.consume("d"))
    assert len(steppers) == 1
    assert steppers[0].get_current_value() == "d"


def test_should_complete_step_with_char_limit():
    """Test should_complete_step with character limits."""
    # Test with max characters
    limited_state_machine = CharacterStateMachine(char_limit=3)
    stepper = CharacterStepper(limited_state_machine, "abc")  # Already at limit
    assert stepper.should_complete_step()

    # Test with exceeding max characters
    stepper = CharacterStepper(limited_state_machine, "abcd")  # Exceeds limit
    assert not stepper.should_complete_step()

    # Test with min characters
    min_state_machine = CharacterStateMachine(char_min=2)
    stepper = CharacterStepper(min_state_machine, "a")  # Below minimum
    assert not stepper.should_complete_step()

    stepper = CharacterStepper(min_state_machine, "ab")  # At minimum
    assert stepper.should_complete_step()


def test_accepts_any_token():
    """Test accepts_any_token method."""
    sm = CharacterStateMachine(whitelist_charset=["a", "b", "c"])
    stepper = CharacterStepper(sm)
    assert not stepper.accepts_any_token()
    assert len(list(stepper.consume("X"))) == 0
    assert sorted(stepper.get_valid_continuations()) == ["a", "b", "c"]

    sm2 = CharacterStateMachine()
    stepper2 = CharacterStepper(sm2)
    assert stepper2.accepts_any_token()
