import pytest
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


def test_whitespace_acceptor_default():
    """Test WhitespaceAcceptor with default settings."""
    state_machine = WhitespaceStateMachine()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "   ")
    ]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "   "


def test_whitespace_acceptor_custom_min_whitespace():
    """Test WhitespaceAcceptor with custom min_whitespace."""
    state_machine = WhitespaceStateMachine(min_whitespace=2)
    walkers = state_machine.get_walkers()
    # Input with one whitespace character, should not be accepted
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, " ")]
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)

    # Input with two whitespace characters, should be accepted
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, "  ")]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_custom_max_whitespace():
    """Test WhitespaceAcceptor with custom max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=3)
    walkers = list(state_machine.get_walkers())

    # Input exceeding max_whitespace, should not be accepted
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "    ")
    ]
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)

    # Input within max_whitespace, should be accepted
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "   ")
    ]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


@pytest.mark.parametrize(
    "token, expected_value",
    [
        (" ", " "),
        ("\t", "\t"),
        ("\n", "\n"),
        ("\r", "\r"),
        (" \t\n\r", " \t\n\r"),
    ],
)
def test_whitespace_acceptor_various_whitespace_characters(token, expected_value):
    """Test WhitespaceAcceptor with different whitespace characters."""
    state_machine = WhitespaceStateMachine()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, token)
    ]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == expected_value


def test_whitespace_acceptor_non_whitespace_input():
    """Test WhitespaceAcceptor with non-whitespace input."""
    state_machine = WhitespaceStateMachine()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "abc")
    ]
    assert len(advanced_walkers) == 0
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)

    state_machine = WhitespaceStateMachine(min_whitespace=1)
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "abc")
    ]
    assert len(advanced_walkers) == 0


def test_whitespace_acceptor_mixed_input():
    """Test WhitespaceAcceptor with mixed whitespace and non-whitespace input."""
    state_machine = WhitespaceStateMachine()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "  abc")
    ]
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)
    # Should accept the whitespace part before the non-whitespace character
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, "  ")]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_exceeds_max_whitespace():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=5)
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "      ")
    ]  # Six spaces
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_long_whitespace_within_max():
    """Test WhitespaceAcceptor with long whitespace within max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=10)
    token = " " * 10
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, token)
    ]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_integration_with_text_acceptor():
    """Test integration of WhitespaceAcceptor with TextAcceptor."""

    class CombinedAcceptor(StateMachine):
        def __init__(self):
            super().__init__(
                {
                    0: [(WhitespaceStateMachine(), 1)],
                    1: [(PhraseStateMachine("hello"), "$")],
                },
                start_state=0,
                end_states=["$"],
            )

    state_machine = CombinedAcceptor()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "   hello")
    ]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_with_no_whitespace():
    """Test WhitespaceAcceptor when there's no whitespace between tokens."""
    state_machine = WhitespaceStateMachine()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, "")]
    # Should not be accepted when min_whitespace > 0
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_advance_after_acceptance():
    """Test advancing walker after it has already been accepted."""
    state_machine = WhitespaceStateMachine()
    walkers = state_machine.get_walkers()
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, "  ")]
    assert len(advanced_walkers) == 1
    accepted_walkers = [
        walker for walker in advanced_walkers if walker.has_reached_accept_state()
    ]
    assert len(accepted_walkers) == 1
    # Try advancing accepted walker
    next_walkers: list[Walker] = []
    for walker in accepted_walkers:
        next_walkers.extend(walker.consume(" "))
    assert len(next_walkers) == 1

    next_walkers: list[Walker] = []
    for walker in advanced_walkers:
        next_walkers.extend(walker.consume("    "))
    assert len(next_walkers) == 1

def test_whitespace_acceptor_state_machine():
    """Test StateMachine with WhitespaceAcceptor and TextAcceptor."""
    sm = StateMachine(
        {
            0: [(WhitespaceStateMachine(), 1)],
            1: [(PhraseStateMachine("hello"), 2)],
            2: [(WhitespaceStateMachine(min_whitespace=1), 3)],
            3: [(PhraseStateMachine("world"), 4)],
            4: [(WhitespaceStateMachine(), "$")],
        },
        start_state=0,
        end_states=["$"],
    )
    walkers = sm.get_walkers()
    assert len(walkers) == 2
    # we expect 2 walker, one for the whitespace and one for the next state_machine (since min_whitespace=0)
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "   hello")
    ]
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "   hello"
    for char in "  world    ":
        advanced_walkers = [
            walker for _, walker in StateMachine.advance_all(advanced_walkers, char)
        ]
    assert len(advanced_walkers) == 1
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_max_whitespace_exceeded():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=5)
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "     ")
    ]  # five spaces
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(advanced_walkers, " ")
    ]  # one more space
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)
