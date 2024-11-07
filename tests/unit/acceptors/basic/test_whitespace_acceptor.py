from lexpy import DAWG
import pytest
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor, WhitespaceWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.json.string_acceptor import StringAcceptor
from pse.acceptors.json.object_acceptor import ObjectAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.walker import Walker


def test_whitespace_acceptor_default():
    """Test WhitespaceAcceptor with default settings."""
    acceptor = WhitespaceAcceptor()
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "   "))
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.current_value() == "   "


def test_whitespace_acceptor_custom_min_whitespace():
    """Test WhitespaceAcceptor with custom min_whitespace."""
    acceptor = WhitespaceAcceptor(min_whitespace=2)
    walkers = list(acceptor.get_walkers())

    # Input with one whitespace character, should not be accepted
    walkers = list(StateMachine.advance_all(walkers, " "))
    print(f"walkers: {walkers}")
    assert not any(walker.has_reached_accept_state() for walker in walkers)

    # Input with two whitespace characters, should be accepted
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "  "))
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_whitespace_acceptor_custom_max_whitespace():
    """Test WhitespaceAcceptor with custom max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=3)
    walkers = list(acceptor.get_walkers())

    # Input exceeding max_whitespace, should not be accepted
    walkers = list(StateMachine.advance_all(walkers, "    "))
    assert not any(walker.has_reached_accept_state() for walker in walkers)

    # Input within max_whitespace, should be accepted
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "   "))
    assert any(walker.has_reached_accept_state() for walker in walkers)


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
    acceptor = WhitespaceAcceptor()
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, token))
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.current_value() == expected_value

def test_whitespace_acceptor_non_whitespace_input():
    """Test WhitespaceAcceptor with non-whitespace input."""
    acceptor = WhitespaceAcceptor()
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "abc"))
    assert len(walkers) == 0
    assert not any(walker.has_reached_accept_state() for walker in walkers)

    acceptor = WhitespaceAcceptor(min_whitespace=1)
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "abc"))
    assert len(walkers) == 0


def test_whitespace_acceptor_mixed_input():
    """Test WhitespaceAcceptor with mixed whitespace and non-whitespace input."""
    acceptor = WhitespaceAcceptor()
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "  abc"))
    print(f"walkers: {walkers}")
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    # Should accept the whitespace part before the non-whitespace character
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "  "))
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_whitespace_acceptor_exceeds_max_whitespace():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=5)
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "      "))  # Six spaces
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_whitespace_acceptor_long_whitespace_within_max():
    """Test WhitespaceAcceptor with long whitespace within max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=10)
    token = " " * 10
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, token))
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_whitespace_acceptor_walker_get_value():
    """Test get_value method of WhitespaceAcceptor.Walker."""
    acceptor = WhitespaceAcceptor()
    walker = WhitespaceWalker(acceptor, text=" \t")
    assert walker.current_value() == " \t"


def test_whitespace_acceptor_walker_is_in_value():
    """Test is_in_value method of WhitespaceAcceptor.Walker."""
    acceptor = WhitespaceAcceptor()
    walker = WhitespaceWalker(acceptor, text="")
    assert not walker.is_within_value()
    walker = WhitespaceWalker(acceptor, text=" ")
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
                initial_state=0,
                end_states=["$"],
            )

    acceptor = CombinedAcceptor()
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "   hello"))
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_whitespace_acceptor_integration_with_object_acceptor():
    """Test WhitespaceAcceptor in the context of ObjectAcceptor."""
    token = '{ "key": "value", "number": 42 }'
    acceptor = ObjectAcceptor()
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, token))

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            obj = walker.current_value()
            assert obj == {"key": "value", "number": 42}


def test_whitespace_acceptor_with_no_whitespace():
    """Test WhitespaceAcceptor when there's no whitespace between tokens."""
    acceptor = WhitespaceAcceptor()
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, ""))
    # Should not be accepted when min_whitespace > 0
    assert not any(walker.has_reached_accept_state() for walker in walkers)

def test_whitespace_acceptor_partial_whitespace_input():
    """Test advancing with partial whitespace followed by non-whitespace."""
    dawg = DAWG()
    dawg.add("  ")
    dawg.add("a")
    acceptor = WhitespaceAcceptor()
    walkers = list(acceptor.get_walkers())
    advancing_token = "  a"
    result = list(StateMachine.advance_all_walkers(walkers, advancing_token, dawg))
    assert len(result) == 1
    for advanced_token, walker in result:
        assert walker.has_reached_accept_state()
        assert advanced_token == "  "
        assert walker.current_value() == "  "




def test_whitespace_acceptor_walker_length_exceeded():
    """Test that walker sets length_exceeded when max_whitespace is exceeded."""
    acceptor = WhitespaceAcceptor(max_whitespace=2)
    walker = WhitespaceWalker(acceptor, text="  ")
    assert not walker.length_exceeded
    walker = WhitespaceWalker(acceptor, text="   ")
    assert walker.length_exceeded


def test_whitespace_acceptor_advance_after_acceptance():
    """Test advancing walker after it has already been accepted."""
    acceptor = WhitespaceAcceptor()
    walkers = acceptor.get_walkers()
    walkers = list(StateMachine.advance_all(walkers, "  "))
    assert len(walkers) == 1
    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert len(accepted_walkers) == 1
    # Try advancing accepted walker
    next_walkers: list[Walker] = []
    for walker in accepted_walkers:
        next_walkers.extend(walker.consume_token(" "))
    assert len(next_walkers) == 1

    next_walkers: list[Walker] = []
    for walker in walkers:
        next_walkers.extend(walker.consume_token("    "))
    assert len(next_walkers) == 1


def test_whitespace_acceptor_no_remaining_input():
    """Test that the walker handles no remaining input correctly."""
    acceptor = WhitespaceAcceptor()
    walker = WhitespaceWalker(acceptor)
    assert walker.remaining_input is None
    walkers = list(walker.consume_token("   "))
    for c in walkers:
        if c.has_reached_accept_state():
            assert c.remaining_input is None


def test_whitespace_acceptor_walker_equality():
    """Test equality and hashing of WhitespaceAcceptor.Walker."""
    acceptor = WhitespaceAcceptor()
    walker1 = WhitespaceWalker(acceptor, text=" ")
    walker2 = WhitespaceWalker(acceptor, text=" ")
    assert walker1 == walker2
    assert hash(walker1) == hash(walker2)


def test_whitespace_acceptor_walker_clone():
    """Test cloning functionality of WhitespaceAcceptor.Walker."""
    acceptor = WhitespaceAcceptor()
    walker = WhitespaceWalker(acceptor, text=" ")
    cloned_walker = walker.clone()
    assert walker == cloned_walker
    assert walker is not cloned_walker
    assert walker.current_value() == cloned_walker.current_value()


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
        initial_state=0,
        end_states=["$"],
    )
    walkers = list(sm.get_walkers())
    assert len(walkers) == 2
    # we expect 2 walker, one for the whitespace and one for the next acceptor (since min_whitespace=0)
    walkers = list(StateMachine.advance_all(walkers, "   hello"))
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.current_value() == "   hello"
    for char in "  world    ":
        walkers = list(StateMachine.advance_all(walkers, char))
    assert len(walkers) == 1
    assert any(walker.has_reached_accept_state() for walker in walkers)


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
    walkers = list(StateMachine.advance_all(walkers, token))
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.current_value() == '"test"   :   '


def test_max_whitespace_exceeded():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=5)
    walkers = list(acceptor.get_walkers())
    walkers = list(StateMachine.advance_all(walkers, "     "))  # five spaces
    assert any(walker.has_reached_accept_state() for walker in walkers)
    walkers = list(StateMachine.advance_all(walkers, " "))  # one more space
    assert not any(walker.has_reached_accept_state() for walker in walkers)
