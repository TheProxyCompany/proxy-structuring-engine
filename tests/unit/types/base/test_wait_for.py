from pse_core.trie import TrieMap

from pse.types.base.phrase import PhraseStateMachine, PhraseStepper
from pse.types.base.wait_for import (
    WaitFor,
    WaitForStepper,
)


def test_default_wait_for_acceptor() -> None:
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitFor(text_acceptor, min_buffer_length=0)

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
    for stepper, advanced_token, healed in state_machine.advance_all(
        steppers, '"*', trie_map
    ):
        assert healed
        value = stepper.get_current_value()
        assert value == '"'
        assert advanced_token == '"'
    assert len(steppers) == 1
    assert not steppers[0].has_reached_accept_state()
