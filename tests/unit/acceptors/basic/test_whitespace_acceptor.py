import pytest
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.json.string_acceptor import StringAcceptor
from pse.acceptors.json.object_acceptor import ObjectAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.cursor import Cursor


def test_whitespace_acceptor_default():
    """Test WhitespaceAcceptor with default settings."""
    acceptor = WhitespaceAcceptor()
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "   "))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "   "


def test_whitespace_acceptor_custom_min_whitespace():
    """Test WhitespaceAcceptor with custom min_whitespace."""
    acceptor = WhitespaceAcceptor(min_whitespace=2)
    cursors = list(acceptor.get_cursors())

    # Input with one whitespace character, should not be accepted
    cursors = list(StateMachine.advance_all(cursors, " "))
    print(f"cursors: {cursors}")
    assert not any(cursor.in_accepted_state() for cursor in cursors)

    # Input with two whitespace characters, should be accepted
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "  "))
    assert any(cursor.in_accepted_state() for cursor in cursors)


def test_whitespace_acceptor_custom_max_whitespace():
    """Test WhitespaceAcceptor with custom max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=3)
    cursors = list(acceptor.get_cursors())

    # Input exceeding max_whitespace, should not be accepted
    cursors = list(StateMachine.advance_all(cursors, "    "))
    assert not any(cursor.in_accepted_state() for cursor in cursors)

    # Input within max_whitespace, should be accepted
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "   "))
    assert any(cursor.in_accepted_state() for cursor in cursors)


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
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, token))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == expected_value


def test_whitespace_acceptor_empty_input():
    """Test WhitespaceAcceptor with empty input and default min_whitespace."""
    acceptor = WhitespaceAcceptor()
    cursors = list(acceptor.get_cursors())
    assert any(cursor.in_accepted_state() for cursor in cursors)
    cursors = list(StateMachine.advance_all(cursors, " "))
    assert any(cursor.in_accepted_state() for cursor in cursors)


def test_whitespace_acceptor_min_whitespace_zero():
    """Test WhitespaceAcceptor with min_whitespace set to zero."""
    acceptor = WhitespaceAcceptor(min_whitespace=0)
    cursors = list(acceptor.get_cursors())
    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == ""


def test_whitespace_acceptor_non_whitespace_input():
    """Test WhitespaceAcceptor with non-whitespace input."""
    acceptor = WhitespaceAcceptor()
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "abc"))
    assert len(cursors) == 1
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(cursor.remaining_input == "abc" for cursor in cursors)

    acceptor = WhitespaceAcceptor(min_whitespace=1)
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "abc"))
    assert len(cursors) == 0


def test_whitespace_acceptor_mixed_input():
    """Test WhitespaceAcceptor with mixed whitespace and non-whitespace input."""
    acceptor = WhitespaceAcceptor()
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "  abc"))
    print(f"cursors: {cursors}")
    assert not any(cursor.in_accepted_state() for cursor in cursors)
    # Should accept the whitespace part before the non-whitespace character
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "  "))
    assert any(cursor.in_accepted_state() for cursor in cursors)


def test_whitespace_acceptor_exceeds_max_whitespace():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=5)
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "      "))  # Six spaces
    assert not any(cursor.in_accepted_state() for cursor in cursors)


def test_whitespace_acceptor_long_whitespace_within_max():
    """Test WhitespaceAcceptor with long whitespace within max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=10)
    token = " " * 10
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, token))
    assert any(cursor.in_accepted_state() for cursor in cursors)


def test_whitespace_acceptor_cursor_get_value():
    """Test get_value method of WhitespaceAcceptor.Cursor."""
    acceptor = WhitespaceAcceptor()
    cursor = acceptor.Cursor(acceptor, text=" \t")
    assert cursor.get_value() == " \t"


def test_whitespace_acceptor_cursor_is_in_value():
    """Test is_in_value method of WhitespaceAcceptor.Cursor."""
    acceptor = WhitespaceAcceptor()
    cursor = acceptor.Cursor(acceptor, text="")
    assert not cursor.is_in_value()
    cursor = acceptor.Cursor(acceptor, text=" ")
    assert cursor.is_in_value()


def test_whitespace_acceptor_expects_more_input():
    """Test that WhitespaceAcceptor does not expect more input."""
    acceptor = WhitespaceAcceptor()
    cursor = acceptor.Cursor(acceptor)
    assert acceptor.expects_more_input(cursor)


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
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "   hello"))
    assert any(cursor.in_accepted_state() for cursor in cursors)


def test_whitespace_acceptor_integration_with_object_acceptor():
    """Test WhitespaceAcceptor in the context of ObjectAcceptor."""
    token = '{ "key": "value", "number": 42 }'
    acceptor = ObjectAcceptor()
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, token))

    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            obj = cursor.get_value()
            assert obj == {"key": "value", "number": 42}


def test_whitespace_acceptor_with_no_whitespace():
    """Test WhitespaceAcceptor when there's no whitespace between tokens."""
    acceptor = WhitespaceAcceptor()
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, ""))
    # Should not be accepted when min_whitespace > 0
    assert not any(cursor.in_accepted_state() for cursor in cursors)


def test_whitespace_acceptor_zero_length_input():
    """Test WhitespaceAcceptor with zero-length input when min_whitespace is zero."""
    acceptor = WhitespaceAcceptor(min_whitespace=0)
    cursors = list(acceptor.get_cursors())

    # Check if any cursor is already in the accepted state
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == ""
            break
    else:
        assert False, "No accepted cursors found before advancing."


def test_whitespace_acceptor_partial_whitespace_input():
    """Test advancing with partial whitespace followed by non-whitespace."""
    acceptor = WhitespaceAcceptor()
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "  a"))
    # Should accept the whitespace part before 'a'
    assert not any(cursor.in_accepted_state() for cursor in cursors)
    assert any(
        cursor.get_value() == "  " and cursor.remaining_input == "a"
        for cursor in cursors
    )


def test_whitespace_acceptor_cursor_length_exceeded():
    """Test that cursor sets length_exceeded when max_whitespace is exceeded."""
    acceptor = WhitespaceAcceptor(max_whitespace=2)
    cursor = acceptor.Cursor(acceptor, text="  ")
    assert not cursor.length_exceeded
    cursor = acceptor.Cursor(acceptor, text="   ")
    assert cursor.length_exceeded


def test_whitespace_acceptor_advance_after_acceptance():
    """Test advancing cursor after it has already been accepted."""
    acceptor = WhitespaceAcceptor()
    cursors = acceptor.get_cursors()
    cursors = list(StateMachine.advance_all(cursors, "  "))
    assert len(cursors) == 1
    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert len(accepted_cursors) == 1
    # Try advancing accepted cursor
    next_cursors: list[Cursor] = []
    for cursor in accepted_cursors:
        next_cursors.extend(cursor.advance(" "))
    assert len(next_cursors) == 1

    next_cursors: list[Cursor] = []
    for cursor in cursors:
        next_cursors.extend(cursor.advance("    "))
    assert len(next_cursors) == 1


def test_whitespace_acceptor_no_remaining_input():
    """Test that the cursor handles no remaining input correctly."""
    acceptor = WhitespaceAcceptor()
    cursor = acceptor.Cursor(acceptor)
    assert cursor.remaining_input is None
    cursors = list(cursor.advance("   "))
    for c in cursors:
        if c.in_accepted_state():
            assert c.remaining_input is None


def test_whitespace_acceptor_cursor_equality():
    """Test equality and hashing of WhitespaceAcceptor.Cursor."""
    acceptor = WhitespaceAcceptor()
    cursor1 = acceptor.Cursor(acceptor, text=" ")
    cursor2 = acceptor.Cursor(acceptor, text=" ")
    assert cursor1 == cursor2
    assert hash(cursor1) == hash(cursor2)


def test_whitespace_acceptor_cursor_clone():
    """Test cloning functionality of WhitespaceAcceptor.Cursor."""
    acceptor = WhitespaceAcceptor()
    cursor = acceptor.Cursor(acceptor, text=" ")
    cloned_cursor = cursor.clone()
    assert cursor == cloned_cursor
    assert cursor is not cloned_cursor
    assert cursor.get_value() == cloned_cursor.get_value()


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
    cursors = list(sm.get_cursors())
    assert len(cursors) == 1
    # we expect 1 cursor, one for the whitespace (since min_whitespace=0, it will be accepted)
    cursors = list(StateMachine.advance_all(cursors, "   hello"))
    assert not any(cursor.in_accepted_state() for cursor in cursors)
    for char in "  world    ":
        cursors = list(StateMachine.advance_all(cursors, char))
    assert len(cursors) == 1
    assert any(cursor.in_accepted_state() for cursor in cursors)


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
    cursors = list(sequence_acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, token))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "test   :   "


def test_max_whitespace_exceeded():
    """Test that WhitespaceAcceptor does not accept input exceeding max_whitespace."""
    acceptor = WhitespaceAcceptor(max_whitespace=5)
    cursors = list(acceptor.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "     "))  # five spaces
    assert any(cursor.in_accepted_state() for cursor in cursors)
    cursors = list(StateMachine.advance_all(cursors, " "))  # one more space
    assert not any(cursor.in_accepted_state() for cursor in cursors)
