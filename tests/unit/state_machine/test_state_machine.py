import pytest

from pse.state_machine.state_machine import StateMachine
from pse.state_machine.empty_transition import EmptyTransition
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.primitive_acceptors import BooleanAcceptor, NullAcceptor
from pse.acceptors.basic.character_acceptors import CharacterAcceptor
from pse.acceptors.basic.number.number_acceptor import NumberAcceptor

@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("true", True),
        ("false", False),
    ],
)
def test_boolean_acceptor(input_str, expected_value):
    """Test StateMachine with BooleanAcceptor accepting 'true' or 'false'."""
    sm = StateMachine(
        graph={0: [(BooleanAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, input_str))

    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == expected_value

@pytest.mark.parametrize(
    "input_str, acceptor_args, expected_value",
    [
        ('"world"', {"text": '"world"'}, 'world'),
        ("data", {"text": "data"}, "data"),
        ('"hello"', {"text": '"hello"'}, 'hello'),
        ("I should use a tool", {"text": "I should use a tool"}, "I should use a tool"),
        ('"chain_of_thought"', {"text": '"chain_of_thought"'}, 'chain_of_thought'),
    ],
)
def test_text_acceptor(input_str, acceptor_args, expected_value):
    """Test StateMachine with TextAcceptor transitions."""
    sm = StateMachine(
        graph={0: [(TextAcceptor(**acceptor_args), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, input_str))

    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == expected_value


def test_state_transitions():
    """Test StateMachine with multiple sequential transitions."""
    sm = StateMachine(
        graph={
            0: [(TextAcceptor("start"), 1)],
            1: [(TextAcceptor("middle"), 2)],
            2: [(TextAcceptor("end"), 3)],
        },
        initial_state=0,
        end_states=[3],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "startmiddleend"))

    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "startmiddleend"


def test_cursor_clone():
    """Test cloning functionality of the StateMachine Cursor."""
    sm = StateMachine(
        graph={0: [(TextAcceptor("clone"), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    for cursor in cursors:
        original_cursor = cursor
        cloned_cursor = original_cursor.clone()

        # Advance the original cursor
        new_cursors = list(StateMachine.advance_all([original_cursor], "clone"))
        for new_cursor in new_cursors:
            assert new_cursor.in_accepted_state()
            assert not cloned_cursor.in_accepted_state()
            assert new_cursor != cloned_cursor


def test_null_acceptor():
    """Test StateMachine with NullAcceptor."""
    sm = StateMachine(
        graph={0: [(NullAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "null"))

    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() is None


def test_invalid_input_characters():
    """Test StateMachine handling of invalid input characters."""
    sm = StateMachine(
        graph={0: [(TextAcceptor("valid"), 1)]},
        initial_state=0,
        end_states=[1],
    )

    invalid_input = "vali$d"  # '$' is an invalid character
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, invalid_input))

    # The input contains an invalid character, so there should be no valid cursors
    assert not any(cursor.in_accepted_state() for cursor in cursors)
    assert len(cursors) == 0


def test_partial_matches():
    """Test StateMachine handling of partial matches."""
    sm = StateMachine(
        graph={0: [(TextAcceptor("complete"), 1)]},
        initial_state=0,
        end_states=[1],
    )

    partial_input = "comp"
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, partial_input))

    # No cursors should be in accepted state since the input is incomplete
    assert not any(cursor.in_accepted_state() for cursor in cursors)


@pytest.mark.parametrize(
    "input_str, expected_value",
    [
        ("catdog", "catdog"),
        ("cardoor", "cardoor"),
    ],
)
def test_advance_all_multiple_states(input_str, expected_value):
    """Test StateMachine.advance_all with multiple current states and transitions."""
    sm = StateMachine(
        graph={
            0: [
                (TextAcceptor("cat"), 1),
                (TextAcceptor("car"), 2),
            ],
            1: [(TextAcceptor("dog"), 3)],
            2: [(TextAcceptor("door"), 3)],
        },
        initial_state=0,
        end_states=[3],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, input_str))

    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == expected_value


def test_advance_all_invalid_input():
    """Test StateMachine.advance_all with invalid input characters."""
    sm = StateMachine(
        graph={0: [(TextAcceptor("hello"), 1)]},
        initial_state=0,
        end_states=[1],
    )

    invalid_input = "hell@"
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, invalid_input))

    # The input contains an invalid character '@', so there should be no valid cursors
    assert not any(cursor.in_accepted_state() for cursor in cursors)
    assert len(cursors) == 0


def test_complex_input():
    """Test StateMachine.advance_all with complex input."""
    sm = StateMachine(
        graph={
            0: [(CharacterAcceptor("{"), 1)],
            1: [(CharacterAcceptor("\n"), 2)],
            2: [(CharacterAcceptor("["), 3)],
        },
        initial_state=0,
        end_states=[3],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "{\n["))

    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "{\n["


def test_number_acceptor():
    """Test StateMachine with NumberAcceptor."""

    sm = StateMachine(
        graph={0: [(NumberAcceptor(), 1)]},
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "123.456"))

    assert any(cursor.in_accepted_state() for cursor in cursors)


def test_number_acceptor_in_state_machine_sequence():
    """Test NumberAcceptor within a StateMachine sequence along with other acceptors."""

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Value: "), 1)],
            1: [(NumberAcceptor(), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    cursors = list(sm.get_cursors())
    input_string = "Value: 42"
    cursors = list(sm.advance_all(cursors, input_string))

    assert any(
        cursor.in_accepted_state() for cursor in cursors
    ), "StateMachine should accept combined text and number input."
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert (
                cursor.get_value() == "Value: 42"
            ), "Parsed value should be the combined string 'Value: 42'."


def test_char_by_char_in_state_machine():
    """Test NumberAcceptor within a StateMachine sequence along with other acceptors."""

    sm = StateMachine(
        graph={
            0: [(TextAcceptor("Value: "), 1)],
            1: [(NumberAcceptor(), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    cursors = list(sm.get_cursors())
    input_string = "Value: 42"
    for char in input_string:
        cursors = list(sm.advance_all(cursors, char))

    assert any(
        cursor.in_accepted_state() for cursor in cursors
    ), "StateMachine should accept combined text and number input."
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert (
                cursor.get_value() == "Value: 42"
            ), "Parsed value should be the combined string 'Value: 42'."


def test_multiple_transitions_with_empty():
    """Test StateMachine with EmptyTransition."""
    sm = StateMachine(
        graph={
            0: [
                (TextAcceptor("start"), 1),
                (EmptyTransition, 2),
            ],
            1: [(TextAcceptor("end"), 3)],
            2: [(TextAcceptor("middle"), 3)],
        },
        initial_state=0,
        end_states=[3],
    )

    # Path with EmptyTransition
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "middle"))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(
        cursor.get_value() == "middle"
        for cursor in cursors
        if cursor.in_accepted_state()
    )

    # Path without EmptyTransition
    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "startend"))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    assert any(
        cursor.get_value() == "startend"
        for cursor in cursors
        if cursor.in_accepted_state()
    )


# Edge case tests


def test_empty_input():
    """Test StateMachine with empty input."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
        },
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    # No input provided, test if cursors are in accepted state
    assert any(cursor.in_accepted_state() for cursor in cursors)


def test_unexpected_input():
    """Test StateMachine with unexpected input."""
    sm = StateMachine(
        graph={
            0: [(TextAcceptor("expected"), 1)],
        },
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "unexpected"))

    # Should not be in accepted state
    assert not any(cursor.in_accepted_state() for cursor in cursors)
    assert len(cursors) == 0


def test_state_machine_no_transitions():
    """Test StateMachine with no transitions from initial state."""
    sm = StateMachine(
        graph={},  # No transitions
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    assert len(cursors) == 0


def test_cursor_class_property():
    """Test the cursor_class property of StateMachine."""
    sm = StateMachine(graph={})
    assert sm.cursor_class is StateMachine.Cursor


def test_advance_cursor_no_transition():
    """Test advance_cursor when cursor has no transition_cursor and not expecting more input."""
    sm = StateMachine(graph={}, initial_state=0, end_states=[])
    cursor = sm.cursor_class(sm)
    cursor.current_state = sm.initial_state
    cursor.transition_cursor = None
    # Ensure in_accepted_state() returns False
    assert not cursor.in_accepted_state()
    # Ensure expects_more_input returns False
    assert not sm.expects_more_input(cursor)
    cursors = list(sm.advance_cursor(cursor, input_str=""))
    # Should not yield any cursors
    assert len(cursors) == 0


def test_state_machine_cursor_get_value_none():
    """Test Cursor.get_value() when there is no transition_cursor and no accept_history."""
    sm = StateMachine(graph={})
    cursor = sm.cursor_class(sm)
    # Ensure transition_cursor and accept_history are None or empty
    cursor.transition_cursor = None
    cursor.accept_history = []
    value = cursor.get_value()
    assert value is None


def test_state_machine_cursor_is_in_value():
    """Test Cursor.is_in_value() when consumed_character_count is 0 and accept_history is empty."""
    sm = StateMachine(graph={})
    cursor = sm.cursor_class(sm)
    cursor.consumed_character_count = 0
    cursor.accept_history = []
    cursor.transition_cursor = None
    assert not cursor.is_in_value()


def test_state_machine_cursor_equality_and_hash():
    """Test Cursor.__eq__ and __hash__ methods."""
    sm = StateMachine(graph={})
    cursor1 = sm.cursor_class(sm)
    cursor1.current_state = 0
    cursor1.target_state = 1

    cursor2 = sm.cursor_class(sm)
    cursor2.current_state = 0
    cursor2.target_state = 1

    assert cursor1 == cursor2
    assert hash(cursor1) == hash(cursor2)

    cursor3 = sm.cursor_class(sm)
    cursor3.current_state = 0
    cursor3.target_state = 2

    assert cursor1 != cursor3
    assert hash(cursor1) != hash(cursor3)


def test_cascade_transition_no_transition_cursor():
    """Test _cascade_transition when cursor.transition_cursor is None."""
    sm = StateMachine(graph={})
    cursor = sm.cursor_class(sm)
    cursor.current_state = 0
    cursor.transition_cursor = None
    cursor.target_state = None

    # Expect an AssertionError due to missing transition_cursor and target_state
    with pytest.raises(AssertionError):
        list(sm._cascade_transition(cursor, visited_states=[], traversed_edges=set()))


def test_find_transitions_no_edges():
    """Test _find_transitions when there are no edges from the current state."""
    sm = StateMachine(graph={}, initial_state=0, end_states=[1])
    cursor = sm.cursor_class(sm)
    cursor.current_state = 0
    transitions = list(
        sm._find_transitions(cursor, visited_states=[], traversed_edges=set())
    )
    # Should be empty since no edges
    assert len(transitions) == 0


def test_get_edges_nonexistent_state():
    """Test get_edges for a state that does not exist in the graph."""
    sm = StateMachine(graph={}, initial_state=0, end_states=[1])
    edges = sm.get_edges(99)  # State 99 does not exist
    assert edges == []


def test_expects_more_input_in_end_state():
    """Test expects_more_input when cursor is in an end state with no remaining input."""
    sm = StateMachine(graph={}, initial_state=0, end_states=[0])
    cursor = sm.cursor_class(sm)
    cursor.current_state = 0
    cursor.remaining_input = ""
    expects_more = sm.expects_more_input(cursor)
    assert not expects_more


def test_state_machine_advance_all_with_no_cursors():
    """Test advance_all when there are no cursors to advance."""
    cursors = []
    advanced_cursors = list(StateMachine.advance_all(cursors, "input"))
    assert advanced_cursors == []


def test_state_machine_empty_transition():
    """Test StateMachine with an EmptyTransition."""
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


def test_state_machine_with_loop():
    """Test StateMachine handling loop transitions."""
    sm = StateMachine(
        graph={
            0: [(TextAcceptor("a"), 0), (TextAcceptor("b"), 1)],
        },
        initial_state=0,
        end_states=[1],
    )

    cursors = list(sm.get_cursors())
    cursors = list(StateMachine.advance_all(cursors, "aaab"))
    assert any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "aaab"


def test_state_machine_advance_cursor_with_remaining_input():
    """Test advance_cursor handling remaining input in transition_cursor."""
    sm = StateMachine(
        graph={
            0: [(TextAcceptor("ab"), 1)],
            1: [(TextAcceptor("cd"), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    initial_cursors = list(sm.get_cursors())
    # Advance with partial input to create remaining input scenario
    cursors = list(StateMachine.advance_all(initial_cursors, "abcde"))
    # Should handle remaining input 'e' after 'abcd'
    assert not any(cursor.in_accepted_state() for cursor in cursors)
    for cursor in cursors:
        if not cursor.in_accepted_state():
            # Remaining input 'e' should be in cursor.remaining_input
            assert cursor.remaining_input == "e"
            assert cursor.get_value() == "abcd"


def test_cursor_equality_and_hash():
    """Test Cursor equality and hashing."""
    sm = StateMachine(graph={})
    cursor1 = sm.cursor_class(sm)
    cursor1.current_state = 0
    cursor1.target_state = 1
    cursor2 = sm.cursor_class(sm)
    cursor2.current_state = 0
    cursor2.target_state = 1
    assert cursor1 == cursor2
    assert hash(cursor1) == hash(cursor2)


def test__cascade_transition_assertion_error():
    """Test that _cascade_transition raises AssertionError when conditions are not met."""
    sm = StateMachine(graph={})
    cursor = sm.cursor_class(sm)
    cursor.current_state = 0
    cursor.transition_cursor = None
    cursor.target_state = None
    with pytest.raises(AssertionError):
        list(sm._cascade_transition(cursor, visited_states=[], traversed_edges=set()))


def test__cascade_transition_complete_transition_false():
    """Test _cascade_transition when complete_transition returns False."""
    sm = StateMachine(graph={})
    cursor = sm.cursor_class(sm)
    cursor.current_state = 0
    cursor.transition_cursor = sm.cursor_class(sm)
    cursor.target_state = 1
    # Mock complete_transition to return False
    cursor.complete_transition = lambda *args, **kwargs: False
    transitions = list(
        sm._cascade_transition(cursor, visited_states=[], traversed_edges=set())
    )
    assert len(transitions) == 0  # Should not proceed if transition is incomplete


def test_advance_cursor_no_transition_and_not_expecting_more():
    """Test advance_cursor when cursor has no transition_cursor and not expecting more input."""
    sm = StateMachine(graph={}, initial_state=0, end_states=[])
    cursor = sm.cursor_class(sm)
    cursor.current_state = sm.initial_state
    cursor.transition_cursor = None
    # Ensure in_accepted_state() returns False
    assert not cursor.in_accepted_state()
    # Ensure expects_more_input returns False
    assert not sm.expects_more_input(cursor)
    cursors = list(sm.advance_cursor(cursor, input_str=""))
    # Should not yield any cursors
    assert len(cursors) == 0
