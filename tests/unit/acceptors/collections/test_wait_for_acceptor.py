from typing import Tuple

import pytest
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor, WaitForWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.core.state_machine import StateMachine


@pytest.fixture
def setup_acceptors() -> Tuple[WaitForAcceptor, TextAcceptor]:
    """
    Fixture to set up common acceptors and the WaitForAcceptor instance for tests.
    """
    trigger_acceptor = TextAcceptor("END")
    wait_for_acceptor = WaitForAcceptor(wait_for_acceptor=trigger_acceptor)
    return wait_for_acceptor, trigger_acceptor


def test_initial_walkers(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that the initial walkers are correctly initialized.
    """
    wait_for_acceptor, _ = setup_acceptors
    walkers = list(wait_for_acceptor.get_walkers())

    assert len(walkers) == 1, "There should be exactly one initial walker."
    assert isinstance(
        walkers[0], WaitForWalker
    ), "Walker should be an instance of WaitForwalker."
    assert not walkers[
        0
    ].has_reached_accept_state(), "Initial walker should not be in an accepted state."


def test_matches_all(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that the Walker's matches_all method always returns True.
    """
    wait_for_acceptor, _ = setup_acceptors
    walker = list(wait_for_acceptor.get_walkers())[0]

    assert walker.accepts_any_token(), "matches_all should always return True."


@pytest.mark.parametrize(
    "input_sequence, expected_triggered",
    [
        ("This is a test without the trigger.", False),
        ("This is a test END", True),
    ],
)
def test_advance_with_and_without_trigger(
    setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor],
    input_sequence: str,
    expected_triggered: bool,
):
    """
    Test advancing characters with and without the trigger.
    """
    wait_for_acceptor, _ = setup_acceptors
    walkers = list(wait_for_acceptor.get_walkers())

    for char in input_sequence:
        walkers = [walker for _, walker in wait_for_acceptor.advance_all(walkers, char)]

    assert (
        any(walker.has_reached_accept_state() for walker in walkers)
        == expected_triggered
    ), f"Walker should {'be' if expected_triggered else 'not be'} in accepted state."
    assert (
        wait_for_acceptor.triggered == expected_triggered
    ), f"WaitForAcceptor should {'be' if expected_triggered else 'not be'} triggered."


def test_get_value_after_trigger(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test the get_value method after the trigger is encountered.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "Triggering the acceptor END"
    walkers = list(wait_for_acceptor.get_walkers())

    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.current_value == "END"
            ), "get_value should return 'END' after trigger."


@pytest.mark.parametrize(
    "input_sequence", ["First END and then another END.", "END immediately."]
)
def test_multiple_triggers(
    setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor], input_sequence: str
):
    """
    Test handling multiple triggers within the input sequence.
    """
    wait_for_acceptor, _ = setup_acceptors
    walkers = list(wait_for_acceptor.get_walkers())

    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    for walker in walkers:
        if walker.has_reached_accept_state():
            assert (
                walker.current_value == "END"
            ), "get_value should return 'END' after trigger."
        else:
            assert (
                walker.current_value == "TextAcceptor('END')"
            ), "get_value should return the waiting state description."

    assert (
        wait_for_acceptor.triggered
    ), "WaitForAcceptor should be triggered after the first trigger is found."


def test_no_trigger_in_input(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that WaitForAcceptor does not accept when the trigger is not present in the input.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "No triggers here."
    walkers = list(wait_for_acceptor.get_walkers())
    walkers = [
        walker for _, walker in wait_for_acceptor.advance_all(walkers, input_sequence)
    ]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "WaitForAcceptor should not accept without the trigger."
    assert (
        not wait_for_acceptor.triggered
    ), "WaitForAcceptor should not be triggered without the trigger."


def test_empty_input(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that providing an empty input does not lead to acceptance.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = ""
    walkers = list(wait_for_acceptor.get_walkers())
    walkers = [
        walker for _, walker in wait_for_acceptor.advance_all(walkers, input_sequence)
    ]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Empty input should not lead to acceptance."
    assert (
        not wait_for_acceptor.triggered
    ), "WaitForAcceptor should not be triggered with empty input."


def test_acceptor_triggered(setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor]):
    """
    Test that the 'triggered' variable in WaitForAcceptor is set to True when the trigger is found.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "END"
    walkers = list(wait_for_acceptor.get_walkers())

    assert (
        not wait_for_acceptor.triggered
    ), "WaitForAcceptor should not be triggered initially."

    walkers = [
        walker for _, walker in wait_for_acceptor.advance_all(walkers, input_sequence)
    ]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Walker should be in accepted state after trigger."
    assert (
        wait_for_acceptor.triggered
    ), "WaitForAcceptor should be triggered after trigger is found."


def test_triggered_var_within_limit(
    setup_acceptors: Tuple[WaitForAcceptor, TextAcceptor],
):
    """
    Test that the 'triggered' variable in WaitForAcceptor remains False until the trigger is found.
    """
    wait_for_acceptor, _ = setup_acceptors
    input_sequence = "EN"
    walkers = list(wait_for_acceptor.get_walkers())

    assert (
        not wait_for_acceptor.triggered
    ), "WaitForAcceptor should not be triggered initially."

    walkers = [
        walker for _, walker in wait_for_acceptor.advance_all(walkers, input_sequence)
    ]

    assert (
        not wait_for_acceptor.triggered
    ), "WaitForAcceptor should not be triggered until the full trigger is found."

    walkers = [walker for _, walker in wait_for_acceptor.advance_all(walkers, "D")]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Walker should be in accepted state after full trigger."
    assert (
        wait_for_acceptor.triggered
    ), "WaitForAcceptor should be triggered after full trigger is found."
