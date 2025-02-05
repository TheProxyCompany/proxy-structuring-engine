import pytest
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper
from pse_core.trie import TrieMap

from pse.base.character import CharacterStateMachine
from pse.base.phrase import PhraseStateMachine, PhraseStepper
from pse.types.boolean import BooleanStateMachine
from pse.types.integer import IntegerStepper
from pse.types.number import IntegerStateMachine, NumberStateMachine
from pse.types.whitespace import WhitespaceStateMachine


def test_basic():
    sm = StateMachine(
        {
            0: [(PhraseStateMachine("hello"), 1)],
            1: [(WhitespaceStateMachine(), 2)],
            2: [(PhraseStateMachine("world!"), "$")],
        }
    )
    steppers = sm.get_steppers()
    assert len(steppers) == 1
    steppers = sm.advance_all_basic(steppers, "hello")
    assert len(steppers) == 2
    steppers = sm.advance_all_basic(steppers, " ")
    assert len(steppers) == 2
    steppers = sm.advance_all_basic(steppers, "world")
    assert len(steppers) == 1
    steppers = sm.advance_all_basic(steppers, "!")
    assert len(steppers) == 1

    assert all(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "token, expected_value",
    [
        ("true", True),
        ("false", False),
    ],
)
def test_boolean_acceptor(token, expected_value):
    """Test StateMachine with BooleanAcceptor accepting 'true' or 'false'."""
    sm = StateMachine(
        state_graph={0: [(BooleanStateMachine(), 1)]},
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, token)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value


@pytest.mark.parametrize(
    "first, second, end, token",
    [
        ("start", "middle", "end", "startmiddleend"),
        ("1", "2", "3", 123),
    ],
)
def test_state_transitions(first, second, end, token):
    """Test StateMachine with multiple sequential transitions."""
    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine(first), 1)],
            1: [(PhraseStateMachine(second), 2)],
            2: [(PhraseStateMachine(end), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, str(token))

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == token


def test_stepper_clone():
    """Test cloning functionality of the StateMachine Stepper."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("clone"), 1)]},
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    for stepper in steppers:
        original_stepper = stepper
        cloned_stepper = original_stepper.clone()

        # Advance the original stepper
        new_steppers = sm.advance_all_basic([original_stepper], "clone")

        for new_stepper in new_steppers:
            assert new_stepper.has_reached_accept_state()
            assert not cloned_stepper.has_reached_accept_state()
            assert new_stepper != cloned_stepper


def test_invalid_input_characters():
    """Test StateMachine handling of invalid input characters."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("valid"), 1)]},
        start_state=0,
        end_states=[1],
    )

    invalid_input = "vali$d"  # '$' is an invalid character
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, invalid_input)

    # The input contains an invalid character, so there should be no valid steppers
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)
    assert len(steppers) == 0


def test_partial_matches():
    """Test StateMachine handling of partial matches."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("complete"), 1)]},
        start_state=0,
        end_states=[1],
    )

    partial_input = "comp"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, partial_input)

    # No steppers should be in accepted state since the input is incomplete
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        assert stepper.get_current_value() == "comp"


@pytest.mark.parametrize(
    "token, expected_value",
    [
        ("catdog", "catdog"),
        ("cardoor", "cardoor"),
    ],
)
def test_advance_all_multiple_states(token, expected_value):
    """Test StateMachine.advance_all_steppers with multiple current states and transitions."""
    sm = StateMachine(
        state_graph={
            0: [
                (PhraseStateMachine("cat"), 1),
                (PhraseStateMachine("car"), 2),
            ],
            1: [(PhraseStateMachine("dog"), 3)],
            2: [(PhraseStateMachine("door"), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, token)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value


def test_advance_all_invalid_input():
    """Test StateMachine.advance_all_steppers with invalid input characters."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("hello"), 1)]},
        start_state=0,
        end_states=[1],
    )

    invalid_input = "hell@"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, invalid_input)

    # The input contains an invalid character '@', so there should be no valid steppers
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)
    assert len(steppers) == 0


def test_complex_input():
    """Test StateMachine.advance_all_steppers with complex input."""
    sm = StateMachine(
        state_graph={
            0: [(CharacterStateMachine("{"), 1)],
            1: [(CharacterStateMachine("\n"), 2)],
            2: [(CharacterStateMachine("["), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "{\n[")

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "{\n["


def test_number_acceptor():
    """Test StateMachine with NumberAcceptor."""

    sm = StateMachine(
        state_graph={0: [(NumberStateMachine(), 1)]},
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "123.456")

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_number_acceptor_in_state_machine_sequence():
    """Test NumberAcceptor within a StateMachine sequence along with other acceptors."""

    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("Value: "), 1)],
            1: [(NumberStateMachine(), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    steppers = sm.get_steppers()
    input_string = "Value: 42"
    steppers = sm.advance_all_basic(steppers, input_string)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StateMachine should accept combined text and number input."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "Value: 42", (
                "Parsed value should be the combined string 'Value: 42'."
            )

    steppers = sm.advance_all_basic(steppers, ".0")
    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Parsed value should be the combined string 'Value: 42.0'."
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "Value: 42.0", (
                "Parsed value should be the combined string 'Value: 42.0'."
            )


def test_char_by_char_in_state_machine():
    """Test NumberAcceptor within a StateMachine sequence along with other acceptors."""

    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("Value: "), 1)],
            1: [(NumberStateMachine(), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    steppers = sm.get_steppers()
    input_string = "Value: 42"
    for char in input_string:
        steppers = sm.advance_all_basic(steppers, char)
        if not steppers:
            break

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StateMachine should accept combined text and number input."
    )

    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "Value: 42", (
                "Parsed value should be the combined string 'Value: 42'."
            )


# Edge case tests


def test_unexpected_input():
    """Test StateMachine with unexpected input."""
    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("expected"), 1)],
        },
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "unexpected")

    # Should not be in accepted state
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)
    assert len(steppers) == 0


def test_get_edges_nonexistent_state():
    """Test get_edges for a state that does not exist in the graph."""
    sm = StateMachine(state_graph={}, start_state=0, end_states=[1])
    edges = sm.get_edges(99)  # StateId 99 does not exist
    assert edges == []


def test_state_machine_advance_all_with_no_steppers():
    """Test advance_all_steppers when there are no steppers to advance."""
    steppers = []
    steppers = StateMachine.advance_all_basic(steppers, "input")
    assert steppers == []


def test_state_machine_empty_transition():
    """Test StateMachine with an EmptyTransition."""
    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("ignored", is_optional=True), 1)],
            1: [(PhraseStateMachine("test"), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "test")
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "test"


def test_state_machine_with_loop():
    """Test StateMachine handling loop transitions."""
    sm = StateMachine(
        state_graph={
            0: [
                (PhraseStateMachine("a"), 0),
                (PhraseStateMachine("b"), 1),
            ],
        },
        start_state=0,
        end_states=[1],
    )

    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "aaab")
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "aaab"


def test_state_machine_advance_stepper_with_remaining_input():
    """Test advance_stepper handling remaining input in sub_stepper."""
    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("ab"), 1)],
            1: [(PhraseStateMachine("cd"), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    initial_steppers = sm.get_steppers()
    # Advance with partial input to create remaining input scenario
    steppers = sm.advance_all_basic(initial_steppers, "abcde")
    # Should handle remaining input 'e' after 'abcd'
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if not stepper.has_reached_accept_state():
            # Remaining input 'e' should be in stepper.remaining_input
            assert stepper.remaining_input == "e"
            assert stepper.get_current_value() == "abcd"


def test_trie_token_healing():
    """Test StateMachine with WhitespaceAcceptor."""

    trie = TrieMap()
    items = [
        ("\n", 1),
        ("\n\n", 2),
        (" ", 3),
        ("(", 4),
        ("(.", 5),
        (")", 6),
    ]
    trie = trie.insert_all(items)
    sm = StateMachine(
        {
            0: [(PhraseStateMachine("("), 1)],
            1: [(WhitespaceStateMachine(), 2)],
            2: [(PhraseStateMachine(")"), "$")],
        }
    )
    steppers = sm.get_steppers()
    new_steppers = []
    for stepper, advanced_token, healed in sm.advance_all(steppers, "(.", trie):
        assert healed
        assert advanced_token == "("
        assert stepper.get_current_value() == "("
        new_steppers.append(stepper)
    steppers = new_steppers

    assert len(steppers) == 2
    steppers = sm.advance_all_basic(steppers, " \n")
    assert len(steppers) == 2
    steppers = sm.advance_all_basic(steppers, "\n )")
    assert len(steppers) == 1
    assert steppers[0].get_current_value() == "( \n\n )"
    assert steppers[0].has_reached_accept_state()


def test_simple_number_acceptor():
    """Test StateMachine with NumberAcceptor."""
    sm = StateMachine(
        {
            0: [
                (PhraseStateMachine("-", is_optional=True), 1),
            ],
            1: [
                (IntegerStateMachine(), "$"),
            ],
        }
    )

    trie = TrieMap()
    items = [
        ("-", 1),
        ("-1", 2),
        ("1", 3),
    ]
    trie = trie.insert_all(items)
    steppers = sm.get_steppers()
    assert len(steppers) == 2
    text_acceptor_stepper = steppers[0]
    assert isinstance(text_acceptor_stepper, Stepper)
    assert isinstance(text_acceptor_stepper.sub_stepper, PhraseStepper)
    integer_acceptor_stepper = steppers[1]
    assert isinstance(integer_acceptor_stepper, Stepper)
    assert isinstance(integer_acceptor_stepper.sub_stepper, IntegerStepper)

    for stepper, advanced_token, healed in sm.advance_all(steppers, "-1.2", trie):
        assert healed
        assert advanced_token == "-1"
        assert stepper.has_reached_accept_state()
        assert stepper.get_current_value() == -1
