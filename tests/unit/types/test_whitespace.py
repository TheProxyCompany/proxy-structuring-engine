import pytest
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.base.phrase import PhraseStateMachine
from pse.types.whitespace import WhitespaceStateMachine


def test_whitespace_acceptor_default():
    """Test WhitespaceAcceptor with default settings."""
    state_machine = WhitespaceStateMachine()
    steppers = list(state_machine.get_steppers())
    steppers = state_machine.advance_all_basic(steppers, "   ")
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "   "


def test_whitespace_acceptor_custom_min_whitespace():
    """Test WhitespaceAcceptor with custom min_whitespace."""
    state_machine = WhitespaceStateMachine(min_whitespace=2)
    steppers = state_machine.get_steppers()
    # Input with one whitespace character, should not be accepted
    advanced_steppers = state_machine.advance_all_basic(steppers, " ")
    assert not any(stepper.has_reached_accept_state() for stepper in advanced_steppers)

    # Input with two whitespace characters, should be accepted
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "  ")
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


def test_whitespace_acceptor_custom_max_whitespace():
    """Test WhitespaceAcceptor with custom max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=3)
    steppers = list(state_machine.get_steppers())

    # Input exceeding max_whitespace, should not be accepted
    advanced_steppers = state_machine.advance_all_basic(steppers, "    ")
    assert not any(stepper.has_reached_accept_state() for stepper in advanced_steppers)

    # Input within max_whitespace, should be accepted
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "   ")
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


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
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, token)
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value


def test_whitespace_acceptor_non_whitespace_input():
    """Test WhitespaceAcceptor with non-whitespace input."""
    state_machine = WhitespaceStateMachine()
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "abc")
    assert len(advanced_steppers) == 0
    assert not any(stepper.has_reached_accept_state() for stepper in advanced_steppers)

    state_machine = WhitespaceStateMachine(min_whitespace=1)
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "abc")
    assert len(advanced_steppers) == 0


def test_whitespace_acceptor_mixed_input():
    """Test WhitespaceAcceptor with mixed whitespace and non-whitespace input."""
    state_machine = WhitespaceStateMachine()
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "  abc")
    assert not any(stepper.has_reached_accept_state() for stepper in advanced_steppers)
    # Should accept the whitespace part before the non-whitespace character
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "  ")
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


def test_whitespace_acceptor_exceeds_max_whitespace():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=5)
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "      ")
    assert not any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


def test_whitespace_acceptor_long_whitespace_within_max():
    """Test WhitespaceAcceptor with long whitespace within max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=10)
    token = " " * 10
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, token)
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


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
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "   hello")
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


def test_whitespace_acceptor_with_no_whitespace():
    """Test WhitespaceAcceptor when there's no whitespace between tokens."""
    state_machine = WhitespaceStateMachine()
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "")
    # Should not be accepted when min_whitespace > 0
    assert not any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


def test_whitespace_acceptor_advance_after_acceptance():
    """Test advancing stepper after it has already been accepted."""
    state_machine = WhitespaceStateMachine()
    steppers = state_machine.get_steppers()
    advanced_steppers = state_machine.advance_all_basic(steppers, "  ")
    assert len(advanced_steppers) == 1
    accepted_steppers = [
        stepper for stepper in advanced_steppers if stepper.has_reached_accept_state()
    ]
    assert len(accepted_steppers) == 1
    # Try advancing accepted stepper
    next_steppers: list[Stepper] = []
    for stepper in accepted_steppers:
        next_steppers.extend(stepper.consume(" "))
    assert len(next_steppers) == 1

    next_steppers: list[Stepper] = []
    for stepper in advanced_steppers:
        next_steppers.extend(stepper.consume("    "))
    assert len(next_steppers) == 1


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
    steppers = sm.get_steppers()
    assert len(steppers) == 2
    # we expect 2 stepper, one for the whitespace and one for the next state_machine (since min_whitespace=0)
    advanced_steppers = sm.advance_all_basic(steppers, "   hello")
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "   hello"
    for char in "  world    ":
        advanced_steppers = sm.advance_all_basic(advanced_steppers, char)
    assert len(advanced_steppers) == 1
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)


def test_max_whitespace_exceeded():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    state_machine = WhitespaceStateMachine(max_whitespace=5)
    steppers = list(state_machine.get_steppers())
    advanced_steppers = state_machine.advance_all_basic(steppers, "     ")
    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)
    advanced_steppers = state_machine.advance_all_basic(advanced_steppers, " ")
    assert not any(stepper.has_reached_accept_state() for stepper in advanced_steppers)
