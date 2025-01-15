import pytest
from pse_core.state_machine import StateMachine
from pse_core.trie import TrieSet
from pse_core.walker import Walker

from pse.state_machines.base.character import CharacterStateMachine
from pse.state_machines.base.phrase import PhraseStateMachine, PhraseWalker
from pse.state_machines.types.boolean import BooleanStateMachine
from pse.state_machines.types.integer import IntegerWalker
from pse.state_machines.types.number import IntegerStateMachine, NumberStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


def test_basic():
    sm = StateMachine(
        {
            0: [(PhraseStateMachine("hello"), 1)],
            1: [(WhitespaceStateMachine(), 2)],
            2: [(PhraseStateMachine("world!"), "$")],
        }
    )
    walkers = sm.get_walkers()
    assert len(walkers) == 1
    walkers = [walker for _, walker in sm.advance_all(walkers, "hello")]
    assert len(walkers) == 2
    walkers = [walker for _, walker in sm.advance_all(walkers, " ")]
    assert len(walkers) == 2
    walkers = [walker for _, walker in sm.advance_all(walkers, "world")]
    assert len(walkers) == 1
    walkers = [walker for _, walker in sm.advance_all(walkers, "!")]
    assert len(walkers) == 1

    assert all(walker.has_reached_accept_state() for walker in walkers)


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

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, token))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == expected_value

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

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, str(token)))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == token


def test_walker_clone():
    """Test cloning functionality of the StateMachine Walker."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("clone"), 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    for walker in walkers:
        original_walker = walker
        cloned_walker = original_walker.clone()

        # Advance the original walker
        advanced = list(StateMachine.advance_all([original_walker], "clone"))
        new_walkers = [w for _, w in advanced]

        for new_walker in new_walkers:
            assert new_walker.has_reached_accept_state()
            assert not cloned_walker.has_reached_accept_state()
            assert new_walker != cloned_walker


def test_invalid_input_characters():
    """Test StateMachine handling of invalid input characters."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("valid"), 1)]},
        start_state=0,
        end_states=[1],
    )

    invalid_input = "vali$d"  # '$' is an invalid character
    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, invalid_input))
    walkers = [walker for _, walker in advanced]

    # The input contains an invalid character, so there should be no valid walkers
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    assert len(walkers) == 0


def test_partial_matches():
    """Test StateMachine handling of partial matches."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("complete"), 1)]},
        start_state=0,
        end_states=[1],
    )

    partial_input = "comp"
    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, partial_input))
    walkers = [walker for _, walker in advanced]

    # No walkers should be in accepted state since the input is incomplete
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        assert walker.get_current_value() == "comp"


@pytest.mark.parametrize(
    "token, expected_value",
    [
        ("catdog", "catdog"),
        ("cardoor", "cardoor"),
    ],
)
def test_advance_all_multiple_states(token, expected_value):
    """Test StateMachine.advance_all_walkers with multiple current states and transitions."""
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

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, token))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == expected_value


def test_advance_all_invalid_input():
    """Test StateMachine.advance_all_walkers with invalid input characters."""
    sm = StateMachine(
        state_graph={0: [(PhraseStateMachine("hello"), 1)]},
        start_state=0,
        end_states=[1],
    )

    invalid_input = "hell@"
    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, invalid_input))
    walkers = [walker for _, walker in advanced]

    # The input contains an invalid character '@', so there should be no valid walkers
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    assert len(walkers) == 0


def test_complex_input():
    """Test StateMachine.advance_all_walkers with complex input."""
    sm = StateMachine(
        state_graph={
            0: [(CharacterStateMachine("{"), 1)],
            1: [(CharacterStateMachine("\n"), 2)],
            2: [(CharacterStateMachine("["), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, "{\n["))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "{\n["


def test_number_acceptor():
    """Test StateMachine with NumberAcceptor."""

    sm = StateMachine(
        state_graph={0: [(NumberStateMachine(), 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, "123.456"))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)


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

    walkers = sm.get_walkers()
    input_string = "Value: 42"
    advanced = list(StateMachine.advance_all(walkers, input_string))
    walkers = [walker for _, walker in advanced]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept combined text and number input."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == "Value: 42"
            ), "Parsed value should be the combined string 'Value: 42'."

    new_advanced = list(StateMachine.advance_all(walkers, ".0"))
    walkers = [walker for _, walker in new_advanced]
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Parsed value should be the combined string 'Value: 42.0'."
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == "Value: 42.0"
            ), "Parsed value should be the combined string 'Value: 42.0'."


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

    walkers = sm.get_walkers()
    input_string = "Value: 42"
    for char in input_string:
        advanced = list(StateMachine.advance_all(walkers, char))
        walkers = [walker for _, walker in advanced]
        if not walkers:
            break

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StateMachine should accept combined text and number input."

    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.get_current_value() == "Value: 42"
            ), "Parsed value should be the combined string 'Value: 42'."


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

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, "unexpected"))
    walkers = [walker for _, walker in advanced]

    # Should not be in accepted state
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    assert len(walkers) == 0


def test_get_edges_nonexistent_state():
    """Test get_edges for a state that does not exist in the graph."""
    sm = StateMachine(state_graph={}, start_state=0, end_states=[1])
    edges = sm.get_edges(99)  # State 99 does not exist
    assert edges == []


def test_state_machine_advance_all_with_no_walkers():
    """Test advance_all_walkers when there are no walkers to advance."""
    walkers = []
    advanced = list(StateMachine.advance_all(walkers, "input"))
    assert advanced == []


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

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, "test"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "test"


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

    walkers = sm.get_walkers()
    advanced = list(StateMachine.advance_all(walkers, "aaab"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "aaab"


def test_state_machine_advance_walker_with_remaining_input():
    """Test advance_walker handling remaining input in transition_walker."""
    sm = StateMachine(
        state_graph={
            0: [(PhraseStateMachine("ab"), 1)],
            1: [(PhraseStateMachine("cd"), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    initial_walkers = sm.get_walkers()
    # Advance with partial input to create remaining input scenario
    advanced = list(StateMachine.advance_all(initial_walkers, "abcde"))
    walkers = [walker for _, walker in advanced]
    # Should handle remaining input 'e' after 'abcd'
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if not walker.has_reached_accept_state():
            # Remaining input 'e' should be in walker.remaining_input
            assert walker.remaining_input == "e"
            assert walker.get_current_value() == "abcd"


def test_trie_token_healing():
    """Test StateMachine with WhitespaceAcceptor."""

    trie = TrieSet()
    keys = ["\n", "\n\n", " ", "(", "(.", ")"]
    trie = trie.insert_all(keys)
    sm = StateMachine(
        {
            0: [(PhraseStateMachine("("), 1)],
            1: [(WhitespaceStateMachine(), 2)],
            2: [(PhraseStateMachine(")"), "$")],
        }
    )
    walkers = sm.get_walkers()
    new_walkers = []
    for advanced_token, walker in StateMachine.advance_all(walkers, "(.", trie):
        assert advanced_token == "("
        assert walker.get_current_value() == "("
        new_walkers.append(walker)

    assert len(new_walkers) == 2

    advancement = StateMachine.advance_all(new_walkers, " \n\n")
    new_walkers = []
    for advanced_token, walker in advancement:
        assert advanced_token == " \n\n"
        assert walker.get_current_value() == "( \n\n"
        new_walkers.append(walker)

    assert len(new_walkers) == 2

    advancement = StateMachine.advance_all(new_walkers, "\n )")
    new_walkers = []
    for advanced_token, walker in advancement:
        assert advanced_token == "\n )"
        assert walker.get_current_value() == "( \n\n\n )"
        assert walker.has_reached_accept_state()
        new_walkers.append(walker)

    assert len(new_walkers) == 1


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

    trie = TrieSet()
    keys = ["-", "-1", "1"]
    trie = trie.insert_all(keys)
    walkers = sm.get_walkers()
    assert len(walkers) == 2
    text_acceptor_walker = walkers[0]
    assert isinstance(text_acceptor_walker, Walker)
    assert isinstance(text_acceptor_walker.transition_walker, PhraseWalker)
    integer_acceptor_walker = walkers[1]
    assert isinstance(integer_acceptor_walker, Walker)
    assert isinstance(integer_acceptor_walker.transition_walker, IntegerWalker)

    for advanced_token, walker in StateMachine.advance_all(walkers, "-1.2", trie):
        assert advanced_token == "-1"
        assert walker.has_reached_accept_state()
        assert walker.get_current_value() == -1
