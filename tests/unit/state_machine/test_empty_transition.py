from pse.state_machine.empty_transition import EmptyTransitionAcceptor, EmptyTransition
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.accepted_state import AcceptedState
from pse.acceptors.basic.text_acceptor import TextAcceptor

def test_empty_transition_acceptor_initialization():
    """Test the initialization of EmptyTransitionAcceptor."""
    acceptor = EmptyTransitionAcceptor({})
    assert isinstance(acceptor, StateMachine)
    assert acceptor.initial_state == 0
    assert acceptor.end_states == ["$"]

def test_empty_transition_acceptor_get_cursors():
    """Test that get_cursors() yields an AcceptedState with an EmptyTransition.Cursor."""
    acceptor = EmptyTransitionAcceptor({})
    cursors = list(acceptor.get_cursors())
    assert len(cursors) == 1
    cursor = cursors[0]
    assert isinstance(cursor, AcceptedState)
    assert isinstance(cursor.accepted_cursor, EmptyTransitionAcceptor.Cursor)
    assert cursor.in_accepted_state()

def test_empty_transition_cursor_methods():
    """Test the methods of EmptyTransitionAcceptor.Cursor."""
    acceptor = EmptyTransitionAcceptor({})
    cursor = acceptor.Cursor(acceptor)
    assert cursor.get_value() == ""
    assert cursor.__repr__() == "EmptyTransition"
    assert cursor.in_accepted_state()
    assert cursor.consumed_character_count == 0
    assert cursor.current_state == acceptor.initial_state

def test_empty_transition_in_state_machine():
    """Test EmptyTransition used within a StateMachine."""

    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
            1: [(TextAcceptor("test"), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "test"))
    assert any(cursor.in_accepted_state() for cursor in cursors)

    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "test"

def test_empty_transition_with_remaining_input():
    """Test that EmptyTransition correctly handles remaining input."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
            1: [(EmptyTransition, 2)],
            2: [(EmptyTransition, 3)],
            3: [],  # End state
        },
        initial_state=0,
        end_states=[3],
    )

    cursors = list(sm.get_cursors())
    assert any(isinstance(cursor, AcceptedState) for cursor in cursors)
    for cursor in cursors:
        assert cursor.in_accepted_state()
        assert cursor.get_value() == ""

def test_empty_transition_does_not_consume_input():
    """Ensure that EmptyTransition does not consume any input."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1), (TextAcceptor("skip"), 2)],
            1: [(TextAcceptor("test"), 3)],
            2: [(TextAcceptor("test"), 3)],
        },
        initial_state=0,
        end_states=[3],
    )

    # Input that should match the path through EmptyTransition
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "test"))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(cursor.get_value() == "test" for cursor in cursors if cursor.in_accepted_state())

    # Input that should match the path without EmptyTransition
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "skiptest"))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(cursor.get_value() == "skiptest" for cursor in cursors if cursor.in_accepted_state())

def test_empty_transition_in_complex_graph():
    """Test EmptyTransition in a complex graph with multiple paths."""

    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1), (TextAcceptor("a"), 2)],
            1: [(EmptyTransition, 2), (TextAcceptor("b"), 3)],
            2: [(TextAcceptor("c"), 4)],
            3: [(TextAcceptor("d"), 4)],
        },
        initial_state=0,
        end_states=[4],
    )

    # Test path with EmptyTransitions: start -> Empty -> Empty -> 'c' -> end
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "c"))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(cursor.get_value() == "c" for cursor in cursors if cursor.in_accepted_state())

    # Test path: start -> 'a' -> 'c' -> end
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "ac"))
    print(cursors)
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(cursor.get_value() == "ac" for cursor in cursors if cursor.in_accepted_state())

    # Test path: start -> Empty -> 'b' -> 'd' -> end
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "bd"))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(cursor.get_value() == "bd" for cursor in cursors if cursor.in_accepted_state())

def test_empty_transition_cursor_clone():
    """Test cloning of EmptyTransition.Cursor."""
    acceptor = EmptyTransitionAcceptor({})
    cursor = acceptor.Cursor(acceptor)
    cloned_cursor = cursor.clone()
    assert cloned_cursor is not cursor
    assert cloned_cursor.__dict__ == cursor.__dict__

def test_empty_transition_cursor_equality():
    """Test equality and hashing of EmptyTransition.Cursor."""
    acceptor = EmptyTransitionAcceptor({})
    cursor1 = acceptor.Cursor(acceptor)
    cursor2 = acceptor.Cursor(acceptor)
    assert cursor1 == cursor2
    assert hash(cursor1) == hash(cursor2)

def test_empty_transition_cursor_in_accepted_state():
    """Ensure EmptyTransition.Cursor reports it's in an accepted state."""
    acceptor = EmptyTransitionAcceptor({})
    cursor = acceptor.Cursor(acceptor)
    assert cursor.in_accepted_state()
    assert cursor.get_value() == ""

def test_empty_transition_acceptor_in_state_machine():
    """Test using EmptyTransitionAcceptor directly in a StateMachine."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransitionAcceptor({}), 1)],
            1: [(EmptyTransitionAcceptor({}), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    cursors = list(sm.get_cursors())
    assert any(isinstance(cursor, AcceptedState) for cursor in cursors)
    for cursor in cursors:
        assert cursor.in_accepted_state()
