from pse_core.trie import TrieMap

from pse.types.base.phrase import PhraseStateMachine, PhraseStepper
from pse.types.base.wait_for import (
    WaitFor,
    WaitForStepper,
)


def test_default_wait_for_acceptor() -> None:
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitFor(text_acceptor, buffer_length=0)

    steppers = list(state_machine.get_steppers())
    assert len(steppers) == 1
    stepper = steppers[0]
    assert isinstance(stepper, WaitForStepper)
    assert stepper.accepts_any_token()
    assert stepper.sub_stepper
    assert stepper.sub_stepper.state_machine == text_acceptor
    assert isinstance(stepper.sub_stepper, PhraseStepper)
    assert not stepper.is_within_value()
    steppers = state_machine.advance_all_basic(steppers, "Hello ")
    assert len(steppers) == 1
    assert steppers[0].is_within_value()
    steppers = state_machine.advance_all_basic(steppers, "World")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


def test_basic_wait_for_acceptor() -> None:
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitFor(text_acceptor)
    steppers = list(state_machine.get_steppers())
    steppers = state_machine.advance_all_basic(steppers, "Hello World")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


def test_interrupted_wait_for_acceptor() -> None:
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitFor(text_acceptor, strict=True)

    steppers = state_machine.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, "Hello ")
    assert len(steppers) == 1
    assert steppers[0].is_within_value()
    steppers = state_machine.advance_all_basic(
        steppers, "I'm gonna mess up the pattern!"
    )
    assert not steppers


def test_wait_for_acceptor_with_break() -> None:
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitFor(text_acceptor, strict=False)
    steppers = list(state_machine.get_steppers())
    steppers = state_machine.advance_all_basic(steppers, "Hello ")
    assert len(steppers) == 1

    steppers = state_machine.advance_all_basic(
        steppers, "I'm gonna mess up the pattern! But i'll still be accepted!"
    )
    assert len(steppers) == 1

    steppers = state_machine.advance_all_basic(steppers, "World")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


def test_wait_for_acceptor_with_partial_match():
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = PhraseStateMachine('"hello"')
    state_machine = WaitFor(text_acceptor)
    steppers = list(state_machine.get_steppers())
    trie_map = TrieMap()
    items = [
        ('"hello', 1),
        ('"', 2),
        ("hello", 3),
        ('"c', 4),
    ]
    trie_map = trie_map.insert_all(items)
    stepper_deltas = state_machine.advance_all(steppers, '"*', trie_map)
    for stepper_delta in stepper_deltas:
        assert stepper_delta.was_healed
        assert stepper_delta.token == '"'
        assert stepper_delta.stepper.get_current_value() == '"'
    assert len(steppers) == 1
    assert not steppers[0].has_reached_accept_state()


def test_get_valid_continuations_buffer_too_short():
    """Test get_valid_continuations when buffer is shorter than min_buffer_length."""
    text_acceptor = PhraseStateMachine("Hello")
    state_machine = WaitFor(text_acceptor, buffer_length=10)

    steppers = list(state_machine.get_steppers())
    assert len(steppers) == 1
    stepper = steppers[0]

    # Buffer is empty, should return empty list
    continuations = stepper.get_valid_continuations()
    assert continuations == []

def test_should_start_step_with_remaining_input():
    """Test should_start_step when remaining_input is not None."""
    text_acceptor = PhraseStateMachine("Hello")
    state_machine = WaitFor(text_acceptor)

    steppers = list(state_machine.get_steppers())
    assert len(steppers) == 1
    stepper = steppers[0]

    # Set remaining_input
    stepper.remaining_input = "something"

    # should_start_step should be False with remaining_input
    assert not stepper.should_start_step("Hello")


def test_consume_with_no_sub_stepper():
    """Test consume method when sub_stepper is None."""
    state_machine = WaitFor(PhraseStateMachine("Hello"))
    stepper = WaitForStepper(state_machine)

    # Force sub_stepper to be None
    stepper.sub_stepper = None

    # consume should return empty list
    result = stepper.consume("any token")
    assert result == []


def test_consume_with_min_buffer_length_negative():
    """Test consume when min_buffer_length is -1 and not within value."""
    text_acceptor = PhraseStateMachine("Hello")
    state_machine = WaitFor(text_acceptor, buffer_length=-1)

    steppers = list(state_machine.get_steppers())
    assert len(steppers) == 1
    stepper = steppers[0]

    # Make sure we're not within a value
    assert not stepper.is_within_value()

    # Try to consume a token that doesn't start the pattern
    # With buffer_length = -1, this should return empty list
    result = stepper.consume("NotHello")
    assert result == []
