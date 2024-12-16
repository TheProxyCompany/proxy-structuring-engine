import pytest
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.basic.boolean_acceptor import BooleanAcceptor
from pse.state_machines.basic.character_acceptor import CharacterAcceptor
from pse.state_machines.basic.integer_acceptor import IntegerWalker
from pse.state_machines.basic.number_acceptor import IntegerAcceptor, NumberAcceptor
from pse.state_machines.basic.text_acceptor import TextAcceptor, TextWalker
from pse.state_machines.basic.whitespace_acceptor import WhitespaceAcceptor


def test_hello_world():
    sm = StateMachine(
        state_graph={
            0: [(TextAcceptor("hello world"), 1)]
        },
        start_state=0,
        end_states=[1],
    )
    one_walker = sm.get_new_walker()
    assert not one_walker.has_reached_accept_state()
    assert one_walker.current_state == 0
    assert one_walker.consumed_character_count == 0
    assert one_walker.state_machine == sm

    walkers = sm.get_walkers()
    advanced = sm.advance_all(walkers, "hello world")
    walkers = [walker for _, walker in advanced]
    breakpoint()

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "hello world"

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
        state_graph={0: [(BooleanAcceptor(), 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, token))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == expected_value


@pytest.mark.parametrize(
    "token, acceptor_args, expected_value",
    [
        ('"world"', {"text": '"world"'}, "world"),
        ("data", {"text": "data"}, "data"),
        ('"hello"', {"text": '"hello"'}, "hello"),
        ("I should use a tool", {"text": "I should use a tool"}, "I should use a tool"),
        ('"chain_of_thought"', {"text": '"chain_of_thought"'}, "chain_of_thought"),
    ],
)
def test_text_acceptor(token, acceptor_args, expected_value):
    """Test StateMachine with TextAcceptor transitions."""
    sm = StateMachine(
        state_graph={0: [(TextAcceptor(**acceptor_args), 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
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
        ("1", "2", "3", "123"),
    ],
)
def test_state_transitions(first, second, end, token):
    """Test StateMachine with multiple sequential transitions."""
    sm = StateMachine(
        state_graph={
            0: [(TextAcceptor(first), 1)],
            1: [(TextAcceptor(second), 2)],
            2: [(TextAcceptor(end), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, token))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert str(walker.get_current_value()) == token


def test_walker_clone():
    """Test cloning functionality of the StateMachine Walker."""
    sm = StateMachine(
        state_graph={0: [(TextAcceptor("clone"), 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
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
        state_graph={0: [(TextAcceptor("valid"), 1)]},
        start_state=0,
        end_states=[1],
    )

    invalid_input = "vali$d"  # '$' is an invalid character
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, invalid_input))
    walkers = [walker for _, walker in advanced]

    # The input contains an invalid character, so there should be no valid walkers
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    assert len(walkers) == 0


def test_partial_matches():
    """Test StateMachine handling of partial matches."""
    sm = StateMachine(
        state_graph={0: [(TextAcceptor("complete"), 1)]},
        start_state=0,
        end_states=[1],
    )

    partial_input = "comp"
    walkers = list(sm.get_walkers())
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
                (TextAcceptor("cat"), 1),
                (TextAcceptor("car"), 2),
            ],
            1: [(TextAcceptor("dog"), 3)],
            2: [(TextAcceptor("door"), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, token))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == expected_value


def test_advance_all_invalid_input():
    """Test StateMachine.advance_all_walkers with invalid input characters."""
    sm = StateMachine(
        state_graph={0: [(TextAcceptor("hello"), 1)]},
        start_state=0,
        end_states=[1],
    )

    invalid_input = "hell@"
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, invalid_input))
    walkers = [walker for _, walker in advanced]

    # The input contains an invalid character '@', so there should be no valid walkers
    assert not any(walker.has_reached_accept_state() for walker in walkers)
    assert len(walkers) == 0


def test_complex_input():
    """Test StateMachine.advance_all_walkers with complex input."""
    sm = StateMachine(
        state_graph={
            0: [(CharacterAcceptor("{"), 1)],
            1: [(CharacterAcceptor("\n"), 2)],
            2: [(CharacterAcceptor("["), 3)],
        },
        start_state=0,
        end_states=[3],
    )

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, "{\n["))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "{\n["


def test_number_acceptor():
    """Test StateMachine with NumberAcceptor."""

    sm = StateMachine(
        state_graph={0: [(NumberAcceptor(), 1)]},
        start_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all(walkers, "123.456"))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_number_acceptor_in_state_machine_sequence():
    """Test NumberAcceptor within a StateMachine sequence along with other acceptors."""

    sm = StateMachine(
        state_graph={
            0: [(TextAcceptor("Value: "), 1)],
            1: [(NumberAcceptor(), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    walkers = list(sm.get_walkers())
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
            0: [(TextAcceptor("Value: "), 1)],
            1: [(NumberAcceptor(), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    walkers = list(sm.get_walkers())
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
            0: [(TextAcceptor("expected"), 1)],
        },
        start_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
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
            0: [(TextAcceptor("ignored", is_optional=True), 1)],
            1: [(TextAcceptor("test"), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    walkers = list(sm.get_walkers())
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
                (TextAcceptor("a"), 0),
                (TextAcceptor("b"), 1),
            ],
        },
        start_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
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
            0: [(TextAcceptor("ab"), 1)],
            1: [(TextAcceptor("cd"), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    initial_walkers = list(sm.get_walkers())
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


def test_whitespace_acceptor():
    """Test StateMachine with WhitespaceAcceptor."""

    # dawg = DAWG()
    # dawg.add("\n")
    # dawg.add("\n\n")
    # dawg.add(" ")
    # dawg.add("{")
    # dawg.add("{.")
    # dawg.add("}")
    sm = StateMachine(
        {
            0: [(TextAcceptor("{"), 1)],
            1: [
                (WhitespaceAcceptor(), 2),
            ],
            2: [(TextAcceptor("}"), "$")],
        }
    )
    original_walkers = list(sm.get_walkers())
    assert len(original_walkers) == 1
    first_walker = next(iter(original_walkers))
    assert isinstance(first_walker, Walker)
    assert first_walker.current_state == sm.start_state
    assert isinstance(first_walker.transition_walker, TextWalker)

    advancement = StateMachine.advance_all(original_walkers, "{.")
    new_walkers = []
    for advanced_token, walker in advancement:
        assert advanced_token == "{"
        if walker.target_state == "$":
            assert walker.get_current_value() == "{"
        else:
            assert walker.get_current_value() == "{"
        new_walkers.append(walker)

    assert len(new_walkers) == 2, "Expected 2 walkers after advancing with '{.'"

    advancement = StateMachine.advance_all(new_walkers, " ")
    new_walkers = []
    for advanced_token, walker in advancement:
        assert advanced_token == " "
        if walker.target_state == "$":
            assert walker.get_current_value() == "{ "
        else:
            assert walker.get_current_value() == "{ "
        new_walkers.append(walker)

    assert len(new_walkers) == 1, "Expected 1 walker after advancing with ' '"

    advancement = StateMachine.advance_all(new_walkers, "\n}")
    new_walkers = []
    for advanced_token, walker in advancement:
        assert advanced_token == "\n}"
        assert walker.get_current_value() == {}
        assert walker.has_reached_accept_state()
        new_walkers.append(walker)

    assert len(new_walkers) == 1


def test_simple_number_acceptor():
    """Test StateMachine with NumberAcceptor."""
    sm = StateMachine(
        {
            0: [
                (TextAcceptor("-", is_optional=True), 1),
            ],
            1: [
                (IntegerAcceptor(), "$"),
            ],
        }
    )

    # dawg = DAWG()
    # dawg.add("-")
    # dawg.add("-1")
    # dawg.add("1")
    walkers = list(sm.get_walkers())
    assert len(walkers) == 2
    text_acceptor_walker = walkers[0]
    assert isinstance(text_acceptor_walker, Walker)
    assert isinstance(text_acceptor_walker.transition_walker, TextWalker)
    integer_acceptor_walker = walkers[1]
    assert isinstance(integer_acceptor_walker, Walker)
    assert isinstance(integer_acceptor_walker.transition_walker, IntegerWalker)

    for advanced_token, walker in StateMachine.advance_all(walkers, "-1.2"):
        assert advanced_token == "-1"
        assert walker.has_reached_accept_state()
        assert walker.get_current_value() == -1
