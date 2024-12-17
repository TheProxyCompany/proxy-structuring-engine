import pytest
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.basic.string_acceptor import StringAcceptor
from pse.state_machines.basic.text_acceptor import TextAcceptor
from pse.state_machines.basic.whitespace_acceptor import (
    WhitespaceAcceptor,
    WhitespaceWalker,
)
from pse.state_machines.collections.sequence_acceptor import SequenceAcceptor
from pse.state_machines.json.object_acceptor import ObjectAcceptor


def test_whitespace_acceptor_default():
    """Test WhitespaceAcceptor with default settings."""
    state_machine = WhitespaceAcceptor()
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
    state_machine = WhitespaceAcceptor(min_whitespace=2)
    walkers = list(state_machine.get_walkers())

    # Input with one whitespace character, should not be accepted
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, " ")]
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)

    # Input with two whitespace characters, should be accepted
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, "  ")]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_custom_max_whitespace():
    """Test WhitespaceAcceptor with custom max_whitespace."""
    state_machine = WhitespaceAcceptor(max_whitespace=3)
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
    state_machine = WhitespaceAcceptor()
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
    state_machine = WhitespaceAcceptor()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "abc")
    ]
    assert len(advanced_walkers) == 0
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)

    state_machine = WhitespaceAcceptor(min_whitespace=1)
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "abc")
    ]
    assert len(advanced_walkers) == 0


def test_whitespace_acceptor_mixed_input():
    """Test WhitespaceAcceptor with mixed whitespace and non-whitespace input."""
    state_machine = WhitespaceAcceptor()
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
    state_machine = WhitespaceAcceptor(max_whitespace=5)
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "      ")
    ]  # Six spaces
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_long_whitespace_within_max():
    """Test WhitespaceAcceptor with long whitespace within max_whitespace."""
    state_machine = WhitespaceAcceptor(max_whitespace=10)
    token = " " * 10
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, token)
    ]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)


def test_whitespace_acceptor_walker_get_value():
    """Test get_value method of WhitespaceAcceptor.Walker."""
    state_machine = WhitespaceAcceptor()
    walker = WhitespaceWalker(state_machine, value=" \t")
    assert walker.get_current_value() == " \t"


def test_whitespace_acceptor_walker_is_in_value():
    """Test is_in_value method of WhitespaceAcceptor.Walker."""
    state_machine = WhitespaceAcceptor()
    walker = WhitespaceWalker(state_machine, value="")
    assert not walker.is_within_value()
    walker = WhitespaceWalker(state_machine, value=" ")
    assert walker.is_within_value()


def test_whitespace_acceptor_integration_with_text_acceptor():
    """Test integration of WhitespaceAcceptor with TextAcceptor."""

    class CombinedAcceptor(StateMachine):
        def __init__(self):
            super().__init__(
                {
                    0: [(WhitespaceAcceptor(), 1)],
                    1: [(TextAcceptor("hello"), "$")],
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


def test_whitespace_acceptor_integration_with_object_acceptor():
    """Test WhitespaceAcceptor in the context of ObjectAcceptor."""
    token = '{ "key": "value", "number": 42 }'
    state_machine = ObjectAcceptor()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, token)
    ]

    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            obj = walker.get_current_value()
            assert obj == {"key": "value", "number": 42}


def test_whitespace_acceptor_with_no_whitespace():
    """Test WhitespaceAcceptor when there's no whitespace between tokens."""
    state_machine = WhitespaceAcceptor()
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [walker for _, walker in StateMachine.advance_all(walkers, "")]
    # Should not be accepted when min_whitespace > 0
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)


# def test_whitespace_acceptor_partial_whitespace_input():
#     """Test advancing with partial whitespace followed by non-whitespace."""
#     dawg = DAWG()
#     dawg.add("  ")
#     dawg.add("a")
#     state_machine = WhitespaceAcceptor()
#     walkers = list(state_machine.get_walkers())
#     advancing_token = "  a"
#     result = list(StateMachine.advance_all(walkers, advancing_token, dawg))
#     assert len(result) == 1
#     for advanced_token, walker in result:
#         assert walker.has_reached_accept_state()
#         assert advanced_token == "  "
#         assert walker.get_current_value() == "  "


def test_whitespace_acceptor_walker_length_exceeded():
    """Test that walker sets length_exceeded when max_whitespace is exceeded."""
    state_machine = WhitespaceAcceptor(max_whitespace=2)
    walker = WhitespaceWalker(state_machine, value="  ")
    assert not walker.length_exceeded
    walker = WhitespaceWalker(state_machine, value="   ")
    assert walker.length_exceeded


def test_whitespace_acceptor_advance_after_acceptance():
    """Test advancing walker after it has already been accepted."""
    state_machine = WhitespaceAcceptor()
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
        next_walkers.extend(walker.consume_token(" "))
    assert len(next_walkers) == 1

    next_walkers: list[Walker] = []
    for walker in advanced_walkers:
        next_walkers.extend(walker.consume_token("    "))
    assert len(next_walkers) == 1


def test_whitespace_acceptor_no_remaining_input():
    """Test that the walker handles no remaining input correctly."""
    state_machine = WhitespaceAcceptor()
    walker = WhitespaceWalker(state_machine)
    assert walker.remaining_input is None
    for c in walker.consume_token("   "):
        if c.has_reached_accept_state():
            assert c.remaining_input is None


def test_whitespace_acceptor_walker_clone():
    """Test cloning functionality of WhitespaceAcceptor.Walker."""
    state_machine = WhitespaceAcceptor()
    walker = WhitespaceWalker(state_machine, value=" ")
    cloned_walker = walker.clone()
    assert walker == cloned_walker
    assert walker is not cloned_walker
    assert walker.get_current_value() == cloned_walker.get_current_value()


def test_whitespace_acceptor_state_machine():
    """Test StateMachine with WhitespaceAcceptor and TextAcceptor."""
    sm = StateMachine(
        {
            0: [(WhitespaceAcceptor(), 1)],
            1: [(TextAcceptor("hello"), 2)],
            2: [(WhitespaceAcceptor(min_whitespace=1), 3)],
            3: [(TextAcceptor("world"), 4)],
            4: [(WhitespaceAcceptor(), "$")],
        },
        start_state=0,
        end_states=["$"],
    )
    walkers = list(sm.get_walkers())
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


def test_whitespace_acceptor_sequence_acceptor():
    """Test WhitespaceAcceptor in the context of SequenceAcceptor."""
    token = '"test"   :   '
    acceptors = [
        StringAcceptor(),
        WhitespaceAcceptor(),
        TextAcceptor(":"),
        WhitespaceAcceptor(),
    ]
    sequence_acceptor = SequenceAcceptor(acceptors)
    walkers = list(sequence_acceptor.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, token)
    ]
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)
    for walker in advanced_walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == '"test"   :   '


def test_max_whitespace_exceeded():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    state_machine = WhitespaceAcceptor(max_whitespace=5)
    walkers = list(state_machine.get_walkers())
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(walkers, "     ")
    ]  # five spaces
    assert any(walker.has_reached_accept_state() for walker in advanced_walkers)
    advanced_walkers = [
        walker for _, walker in StateMachine.advance_all(advanced_walkers, " ")
    ]  # one more space
    assert not any(walker.has_reached_accept_state() for walker in advanced_walkers)
