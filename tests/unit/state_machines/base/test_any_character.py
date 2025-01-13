"""Unit tests for AnyCharacterStateMachine and AnyCharacterWalker."""

import pytest

from pse.state_machines.base.any_character import (
    AnyCharacterStateMachine,
    AnyCharacterWalker,
)


@pytest.fixture
def basic_state_machine():
    """Fixture for a basic AnyCharacterStateMachine with no restrictions."""
    return AnyCharacterStateMachine()


@pytest.fixture
def allowed_charset_state_machine():
    """Fixture for AnyCharacterStateMachine with allowed charset."""
    return AnyCharacterStateMachine(allowed_charset=["a", "b", "c"])


@pytest.fixture
def disallowed_charset_state_machine():
    """Fixture for AnyCharacterStateMachine with disallowed charset."""
    return AnyCharacterStateMachine(disallowed_charset=["x", "y", "z"])


@pytest.fixture
def case_sensitive_state_machine():
    """Fixture for case-sensitive AnyCharacterStateMachine."""
    return AnyCharacterStateMachine(
        allowed_charset=["A", "B", "C"], case_sensitive=True
    )


@pytest.fixture
def limited_state_machine():
    """Fixture for AnyCharacterStateMachine with character limits."""
    return AnyCharacterStateMachine(char_min=2, char_limit=4)


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("hello", "hello"),
        ("123", "123"),
        ("!@#", "!@#"),
        ("", None),
    ],
)
def test_basic_character_acceptance(basic_state_machine, input_str, expected_value):
    """Test basic character acceptance with no restrictions."""
    walker = AnyCharacterWalker(basic_state_machine)
    walkers = list(walker.consume(input_str))

    if not walkers:
        assert expected_value is None
        return

    assert walkers[0].get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("abc", "abc"),
        ("cab", "cab"),
        ("abcd", None),
        ("xyz", None),
    ],
)
def test_allowed_charset(allowed_charset_state_machine, input_str, expected_value):
    """Test character acceptance with allowed charset."""
    walker = AnyCharacterWalker(allowed_charset_state_machine)
    walkers = list(walker.consume(input_str))

    if not walkers:
        assert expected_value is None
        return

    assert walkers[0].get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("abc", "abc"),
        ("def", "def"),
        ("xyz", None),
        ("axy", None),
    ],
)
def test_disallowed_charset(
    disallowed_charset_state_machine, input_str, expected_value
):
    """Test character acceptance with disallowed charset."""
    walker = AnyCharacterWalker(disallowed_charset_state_machine)
    walkers = list(walker.consume(input_str))

    if not walkers:
        assert expected_value is None
        return

    assert walkers[0].get_current_value() == expected_value


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("ABC", "ABC"),
        ("abc", None),
        ("AbC", None),
    ],
)
def test_case_sensitivity(case_sensitive_state_machine, input_str, expected_value):
    """Test case-sensitive character acceptance."""
    walker = AnyCharacterWalker(case_sensitive_state_machine)
    walkers = list(walker.consume(input_str))

    if not walkers:
        assert expected_value is None
        return

    assert walkers[0].get_current_value() == expected_value


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
def test_character_limits(limited_state_machine, input_str, should_accept):
    """Test character length limits."""
    walker = AnyCharacterWalker(limited_state_machine)
    walkers = list(walker.consume(input_str))

    if should_accept:
        assert len(walkers) == 1
        assert walkers[0].get_current_value() == input_str
    else:
        assert not walkers


def test_walker_clone(basic_state_machine):
    """Test walker cloning functionality."""
    walker = AnyCharacterWalker(basic_state_machine, "test")
    cloned = walker.clone()

    assert walker.get_current_value() == cloned.get_current_value()
    assert walker is not cloned

    # Verify independent advancement
    list(walker.consume("a"))
    assert walker.get_current_value() == "testa"
    assert cloned.get_current_value() == "test"


def test_walker_is_within_value(basic_state_machine):
    """Test is_within_value method."""
    walker = AnyCharacterWalker(basic_state_machine)
    assert walker.is_within_value()

    list(walker.consume("test"))
    assert walker.is_within_value()


def test_should_start_transition(basic_state_machine):
    """Test should_start_transition method."""
    walker = AnyCharacterWalker(basic_state_machine)

    assert walker.should_start_transition("a")
    assert walker.should_start_transition("1")
    assert walker.should_start_transition("!")
    assert not walker.should_start_transition("")


def test_optional_state_machine():
    """Test optional AnyCharacterStateMachine."""
    sm = AnyCharacterStateMachine(is_optional=True)
    walker = AnyCharacterWalker(sm)

    # Should accept empty input
    assert walker.get_current_value() is None

    # Should also accept valid input
    walkers = list(walker.consume("a"))
    assert len(walkers) == 1
    assert walkers[0].get_current_value() == "a"
