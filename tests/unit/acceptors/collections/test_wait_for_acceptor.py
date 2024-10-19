from typing import Tuple

import pytest
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.state_machine.state_machine import StateMachine

@pytest.fixture
def setup_acceptors() -> Tuple[WaitForAcceptor, TextAcceptor]:
    """
    Fixture to set up common acceptors and the WaitForAcceptor instance for tests.
    """
    trigger_acceptor = TextAcceptor("END")
    wait_for_acceptor = WaitForAcceptor(wait_for_acceptor=trigger_acceptor)
    return wait_for_acceptor, trigger_acceptor

def test_initial_cursors(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that the initial cursors are correctly initialized.
    """
    wait_for_acceptor, _ = setup_acceptors
    cursors = list(wait_for_acceptor.get_cursors())

    assert len(cursors) == 1, "There should be exactly one initial cursor."
    assert isinstance(cursors[0], WaitForAcceptor.Cursor), "Cursor should be an instance of WaitForAcceptor.Cursor."
    assert not cursors[0].in_accepted_state(), "Initial cursor should not be in an accepted state."

def test_matches_all(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that the Cursor's matches_all method always returns True.
    """
    wait_for_acceptor, _ = setup_acceptors
    cursor = list(wait_for_acceptor.get_cursors())[0]

    assert cursor.matches_all(), "matches_all should always return True."

@pytest.mark.parametrize("input_sequence, expected_triggered", [
    ("This is a test without the trigger.", False),
    ("This is a test END", True),
])
def test_advance_with_and_without_trigger(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor], input_sequence: str, expected_triggered: bool):
    """
    Test advancing characters with and without the trigger.
    """
    wait_for_acceptor, _ = setup_acceptors
    cursors = list(wait_for_acceptor.get_cursors())

    for char in input_sequence:
        cursors = wait_for_acceptor.advance_all(cursors, char)

    assert any(cursor.in_accepted_state() for cursor in cursors) == expected_triggered, \
        f"Cursor should {'be' if expected_triggered else 'not be'} in accepted state."
    assert wait_for_acceptor.triggered == expected_triggered, \
        f"WaitForAcceptor should {'be' if expected_triggered else 'not be'} triggered."

def test_get_value_before_trigger(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test the get_value method before the trigger is encountered.
    """
    wait_for_acceptor, trigger_acceptor = setup_acceptors
    input_sequence = "Waiting for trigger..."
    cursors = list(wait_for_acceptor.get_cursors())

    for char in input_sequence:
        cursors = StateMachine.advance_all(cursors, char)

    for cursor in cursors:
        if not cursor.in_accepted_state():
            expected_value = str(trigger_acceptor)
            assert cursor.get_value() == expected_value, "get_value should return the correct waiting state description."

def test_get_value_after_trigger(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test the get_value method after the trigger is encountered.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "Triggering the acceptor END"
    cursors = list(wait_for_acceptor.get_cursors())

    for char in input_sequence:
        cursors = StateMachine.advance_all(cursors, char)

    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "END", "get_value should return 'END' after trigger."

@pytest.mark.parametrize("input_sequence", [
    "First END and then another END.",
    "END immediately."
])
def test_multiple_triggers(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor], input_sequence: str):
    """
    Test handling multiple triggers within the input sequence.
    """
    wait_for_acceptor, _ = setup_acceptors
    cursors = list(wait_for_acceptor.get_cursors())

    for char in input_sequence:
        cursors = StateMachine.advance_all(cursors, char)

    for cursor in cursors:
        if cursor.in_accepted_state():
            assert cursor.get_value() == "END", "get_value should return 'END' after trigger."
        else:
            assert cursor.get_value() == "TextAcceptor('END')", "get_value should return the waiting state description."

    assert wait_for_acceptor.triggered, "WaitForAcceptor should be triggered after the first trigger is found."

def test_no_trigger_in_input(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that WaitForAcceptor does not accept when the trigger is not present in the input.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "No triggers here."
    cursors = list(wait_for_acceptor.get_cursors())
    cursors = wait_for_acceptor.advance_all(cursors, input_sequence)

    assert not any(cursor.in_accepted_state() for cursor in cursors), "WaitForAcceptor should not accept without the trigger."
    assert not wait_for_acceptor.triggered, "WaitForAcceptor should not be triggered without the trigger."

def test_empty_input(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that providing an empty input does not lead to acceptance.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = ""
    cursors = list(wait_for_acceptor.get_cursors())
    cursors = wait_for_acceptor.advance_all(cursors, input_sequence)

    assert not any(cursor.in_accepted_state() for cursor in cursors), "Empty input should not lead to acceptance."
    assert not wait_for_acceptor.triggered, "WaitForAcceptor should not be triggered with empty input."

def test_acceptor_triggered(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that the 'triggered' variable in WaitForAcceptor is set to True when the trigger is found.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "END"
    cursors = list(wait_for_acceptor.get_cursors())

    assert not wait_for_acceptor.triggered, "WaitForAcceptor should not be triggered initially."

    cursors = wait_for_acceptor.advance_all(cursors, input_sequence)

    assert any(cursor.in_accepted_state() for cursor in cursors), "Cursor should be in accepted state after trigger."
    assert wait_for_acceptor.triggered, "WaitForAcceptor should be triggered after trigger is found."

def test_triggered_var_within_limit(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that the 'triggered' variable in WaitForAcceptor remains False until the trigger is found.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "EN"
    cursors = list(wait_for_acceptor.get_cursors())

    assert not wait_for_acceptor.triggered, "WaitForAcceptor should not be triggered initially."

    cursors = wait_for_acceptor.advance_all(cursors, input_sequence)

    assert not wait_for_acceptor.triggered, "WaitForAcceptor should not be triggered until the full trigger is found."

    cursors = wait_for_acceptor.advance_all(cursors, "D")

    assert any(cursor.in_accepted_state() for cursor in cursors), "Cursor should be in accepted state after full trigger."
    assert wait_for_acceptor.triggered, "WaitForAcceptor should be triggered after full trigger is found."
