"""Unit tests for the CharacterAcceptor and its integration with the StateMachine."""

from __future__ import annotations

import pytest
from pse_core.state_machine import StateMachine
from pse_core.trie import TrieSet

from pse.state_machines.base.character import (
    CharacterStateMachine,
    CharacterStepper,
)


@pytest.mark.parametrize(
    "charset, input_string, expected_value, should_accept",
    [
        (["a", "b", "c"], "abc", "abc", True),
        (["x", "y", "z"], "abc", None, False),
        (["1", "2", "3"], "123", 123, True),
        (["a", "b", "c"], "AbC", "abc", True),  # Case-insensitive test
        ([], "anything", None, False),  # Empty charset
        (["あ", "い", "う"], "あいう", "あいう", True),  # Non-ASCII characters
        ([" ", "\t", "\n"], " \t\n", " \t\n", True),  # Whitespace characters
    ],
)
def test_character_acceptor_basic(
    charset: list[str],
    input_string: str,
    expected_value: str | None,
    should_accept: bool,
) -> None:
    """
    Test CharacterAcceptor with various charsets and input strings.

    Args:
        charset: Set of characters to accept
        input_string: String to test
        expected_value: Expected output value
        should_accept: Whether input should be accepted
        state_machine_factory: Factory function for creating StateMachine
    """
    sm = StateMachine(
        state_graph={0: [(CharacterStateMachine(charset, case_sensitive=False), 1)]},
        start_state=0,
        end_states=[1],
    )

    advanced = StateMachine.advance_all_basic(sm.get_steppers(), input_string)
    for stepper in advanced:
        if should_accept:
            assert stepper.has_reached_accept_state()
            assert stepper.get_current_value() == expected_value
        else:
            assert not stepper.has_reached_accept_state()


@pytest.mark.parametrize(
    "charset, char_limit, input_string, expected_value",
    [
        (["1", "2", "3"], 2, "123", 12),
        (["a", "b", "c"], 1, "abc", "a"),
        (["x", "y", "z"], 3, "xy", "xy"),
    ],
)
def test_character_acceptor_char_limit(
    charset: list[str],
    char_limit: int,
    input_string: str,
    expected_value: str,
) -> None:
    """
    Test CharacterAcceptor with character limits.

    Args:
        charset: Set of characters to accept
        char_limit: Maximum number of characters to accept
        input_string: String to test
        expected_value: Expected output value
        state_machine_factory: Factory function for creating StateMachine
    """
    sm = StateMachine(
        state_graph={0: [(CharacterStateMachine(charset, char_limit=char_limit), 1)]},
        start_state=0,
        end_states=[1],
    )

    trie = TrieSet()
    trie.insert_all([str(expected_value), str(input_string)])

    steppers = sm.get_steppers()
    steppers = StateMachine.advance_all(steppers, input_string, trie, token_healing=True)

    assert len(steppers) == 1, "Expected 1 stepper after advancing"
    assert any(stepper.has_reached_accept_state() for stepper, _, _ in steppers)
    for stepper, _, _ in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value


def test_character_acceptor_select() -> None:
    """Test the select method of CharacterStepper."""
    charset = ["a", "b", "c"]
    state_machine = CharacterStateMachine(charset)
    stepper = CharacterStepper(state_machine)

    trie = TrieSet()
    trie.insert_all(charset)

    selections = stepper.get_valid_continuations()
    assert set(selections) == set(charset)
