import pytest
from pse.state_machine.empty_transition import EmptyTransition, EmptyTransitionAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.number.integer_acceptor import IntegerAcceptor


def test_empty_transition_acceptor_initialization():
    """Test the initialization of EmptyTransitionAcceptor."""
    acceptor = EmptyTransitionAcceptor({})
    assert isinstance(acceptor, StateMachine)
    assert acceptor.initial_state == 0
    assert acceptor.end_states == ["$"]


def test_empty_transition_acceptor_get_walkers():
    """Test that get_walkers() yields no walkers since EmptyTransition does not consume input."""
    acceptor = EmptyTransitionAcceptor({})
    walkers = list(acceptor.get_walkers())
    assert len(walkers) == 0


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

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "test"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)

    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.accumulated_value() == "test"


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

    walkers = list(sm.get_walkers())
    # Since EmptyTransition does not consume input, there is no input to advance
    # We directly check if any walkers have reached the accept state
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.accumulated_value() is None


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
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "test"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "test"
        for walker in walkers
        if walker.has_reached_accept_state()
    )

    # Input that should match the path without EmptyTransition
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "skiptest"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "skiptest"
        for walker in walkers
        if walker.has_reached_accept_state()
    )


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
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "c"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "c"
        for walker in walkers
        if walker.has_reached_accept_state()
    )

    # Test path: start -> 'a' -> 'c' -> end
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "ac"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "ac"
        for walker in walkers
        if walker.has_reached_accept_state()
    )

    # Test path: start -> Empty -> 'b' -> 'd' -> end
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "bd"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "bd"
        for walker in walkers
        if walker.has_reached_accept_state()
    )


def test_empty_transition_with_cycle():
    """Test StateMachine with a cycle involving EmptyTransition."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 0), (TextAcceptor("end"), 1)],
        },
        initial_state=0,
        end_states=[1],
    )

    # Input that should eventually reach the end state
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "end"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "end"
        for walker in walkers
        if walker.has_reached_accept_state()
    )


def test_multiple_empty_transitions_to_same_state():
    """Test multiple EmptyTransitions leading to the same state."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1), (EmptyTransition, 1)],
            1: [(TextAcceptor("test"), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "test"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_empty_transition_followed_by_required_input():
    """Test EmptyTransition followed by an acceptor that requires input."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
            1: [(IntegerAcceptor(), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    # Input that matches the IntegerAcceptor
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "123"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.accumulated_value() == 123


def test_empty_transition_at_end_state():
    """Test EmptyTransition leading directly to an end state."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
        },
        initial_state=0,
        end_states=[1],
    )

    walkers = list(sm.get_walkers())
    # No input, check if walkers reach accept state
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_empty_transition_with_no_end_state():
    """Test StateMachine with EmptyTransition but no possible end state."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
            1: [(EmptyTransition, 0)],  # Loop back to state 0
        },
        initial_state=0,
        end_states=[2],  # End state is unreachable
    )

    walkers = list(sm.get_walkers())
    # Since there's no valid path to an end state, there should be no accepted walkers
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_empty_transition_with_optional_transition():
    """Test EmptyTransition interacting with an optional transition."""

    class OptionalTextAcceptor(TextAcceptor):
        def is_optional(self):
            return True

    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1), (OptionalTextAcceptor("opt"), 1)],
            1: [(TextAcceptor("end"), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    # Test path with EmptyTransition
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "end"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)

    # Test path with OptionalTextAcceptor
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "optend"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_empty_transition_precedence():
    """Test that EmptyTransition does not override other valid transitions."""
    sm = StateMachine(
        graph={
            0: [(TextAcceptor("start"), 1), (EmptyTransition, 1)],
            1: [(TextAcceptor("middle"), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    # Input that should take the TextAcceptor path
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "startmiddle"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "startmiddle"
        for walker in walkers
        if walker.has_reached_accept_state()
    )

    # Input that should take the EmptyTransition path
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "middle"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    assert any(
        walker.accumulated_value() == "middle"
        for walker in walkers
        if walker.has_reached_accept_state()
    )


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ("", ""),
        ("test", "test"),
        ("test123", "test123"),
    ],
)
def test_empty_transition_with_various_inputs(input_string, expected_value):
    """Parameterized test for EmptyTransition with various inputs."""
    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
            1: [(TextAcceptor(expected_value), 2)],
        },
        initial_state=0,
        end_states=[2],
    )

    walkers = list(sm.get_walkers())
    if input_string:
        advanced = list(StateMachine.advance_all_walkers(walkers, input_string))
        walkers = [walker for _, walker in advanced]

    has_accepted = any(walker.has_reached_accept_state() for walker in walkers)
    if expected_value == "":
        # Since EmptyTransition does not consume input, input_string should be empty
        assert has_accepted == (input_string == "")
    else:
        assert has_accepted
        assert any(
            walker.accumulated_value() == expected_value
            for walker in walkers
            if walker.has_reached_accept_state()
        )


def test_empty_transition_before_optional_acceptor():
    """Test EmptyTransition before an optional acceptor."""

    class OptionalIntegerAcceptor(IntegerAcceptor):
        def is_optional(self):
            return True

    sm = StateMachine(
        graph={
            0: [(EmptyTransition, 1)],
            1: [(OptionalIntegerAcceptor(), 2)],
            2: [(TextAcceptor("end"), 3)],
        },
        initial_state=0,
        end_states=[3],
    )

    # Test with integer present
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "123end"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    # Since IntegerAcceptor accumulates the integer value, we need to handle that
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert "123end" in str(walker.accumulated_value())

    # Test without integer
    walkers = list(sm.get_walkers())
    advanced = list(StateMachine.advance_all_walkers(walkers, "end"))
    walkers = [walker for _, walker in advanced]
    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.accumulated_value() == "end"
