"""Unit tests for the CharacterAcceptor and its integration with the StateMachine."""

from __future__ import annotations
from typing import Callable, Iterable, Optional

import pytest
from lexpy import DAWG

from pse.acceptors.basic.character_acceptor import CharacterAcceptor, CharacterWalker
from pse.core.state_machine import StateMachine


@pytest.fixture
def state_machine_factory() -> Callable[[CharacterAcceptor], StateMachine]:
    """
    Fixture providing a factory function to create StateMachines with CharacterAcceptors.

    Returns:
        Callable that creates a StateMachine with the given CharacterAcceptor.
    """

    def create(acceptor: CharacterAcceptor) -> StateMachine:
        return StateMachine(
            state_graph={0: [(acceptor, 1)]},
            start_state=0,
            end_states=[1],
        )

    return create


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
    charset: Iterable[str],
    input_string: str,
    expected_value: Optional[str],
    should_accept: bool,
    state_machine_factory: Callable[[CharacterAcceptor], StateMachine],
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
    acceptor = CharacterAcceptor(charset, case_sensitive=False)
    sm = state_machine_factory(acceptor)

    dawg = DAWG()
    dawg.add(input_string)

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, input_string, dawg))

    if should_accept:
        assert any(
            walker.has_reached_accept_state() for _, walker in advanced
        ), f"Should accept '{input_string}'"
        for _, walker in advanced:
            if walker.has_reached_accept_state():
                assert walker.current_value == expected_value
    else:
        assert not any(walker.has_reached_accept_state() for _, walker in advanced)


@pytest.mark.parametrize(
    "charset, char_limit, input_string, expected_value",
    [
        (["1", "2", "3"], 2, "123", "12"),
        (["a", "b", "c"], 1, "abc", "a"),
        (["x", "y", "z"], 3, "xy", "xy"),
    ],
)
def test_character_acceptor_char_limit(
    charset: Iterable[str],
    char_limit: int,
    input_string: str,
    expected_value: str,
    state_machine_factory: Callable[[CharacterAcceptor], StateMachine],
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
    acceptor = CharacterAcceptor(charset, char_limit=char_limit)
    sm = state_machine_factory(acceptor)

    dawg = DAWG()
    dawg.add(expected_value)
    dawg.add(input_string)

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, input_string, dawg))

    assert len(advanced) == 1, "Expected 1 walker after advancing"
    assert any(walker.has_reached_accept_state() for _, walker in advanced)
    for _, walker in advanced:
        if walker.has_reached_accept_state():
            assert str(walker.current_value) == expected_value


def test_character_acceptor_select() -> None:
    """Test the select method of CharacterWalker."""
    charset = ["a", "b", "c"]
    acceptor = CharacterAcceptor(charset)
    walker = CharacterWalker(acceptor)

    dawg = DAWG()
    for char in charset:
        dawg.add(char)

    selections = walker.get_valid_continuations()
    assert set(selections) == set(charset)
