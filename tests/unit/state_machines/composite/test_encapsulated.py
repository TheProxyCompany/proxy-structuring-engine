from __future__ import annotations

import pytest

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.encapsulated import EncapsulatedStateMachine
from pse.state_machines.composite.wait_for import WaitFor
from pse.state_machines.types.object import ObjectStateMachine
from pse.state_machines.types.string import StringStateMachine


@pytest.fixture
def default_delimiters() -> tuple[str, str]:
    """Fixture for default delimiters."""
    return "```json\n", "\n```"

def test_initialization_with_custom_delimiters():
    """Test initializing EncapsulatedAcceptor with custom delimiters."""
    custom_open = "<start>"
    custom_close = "<end>"
    encapsulated_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("content"),
        delimiters=(custom_open, custom_close),
    )

    transitions_state_0 = encapsulated_acceptor.state_graph[0]
    acceptor_0, _ = transitions_state_0[0]
    assert isinstance(acceptor_0, WaitFor)
    assert isinstance(acceptor_0.wait_for_sm, PhraseStateMachine)
    assert acceptor_0.wait_for_sm.phrase == custom_open

    transitions_state_2 = encapsulated_acceptor.state_graph[2]
    acceptor_2, _ = transitions_state_2[0]
    assert isinstance(acceptor_2, PhraseStateMachine)
    assert acceptor_2.phrase == custom_close


def test_accepting_valid_sequence(default_delimiters: tuple[str, str]) -> None:
    """Test accepting a valid sequence with delimiters and content."""
    encapsulated_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("content"),
        delimiters=default_delimiters,
    )
    input_sequence = default_delimiters[0] + "content" + default_delimiters[1]
    steppers = encapsulated_acceptor.get_steppers()
    steppers = encapsulated_acceptor.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_rejecting_sequence_missing_open_delimiter(default_delimiters: tuple[str, str]) -> None:
    """Test rejecting sequence missing the open delimiter."""
    encapsulated_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("content"),
        delimiters=default_delimiters,
    )
    input_sequence = "content" + default_delimiters[1]
    steppers = encapsulated_acceptor.get_steppers()
    steppers = encapsulated_acceptor.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_rejecting_sequence_missing_close_delimiter(default_delimiters: tuple[str, str]) -> None:
    """Test rejecting sequence missing the close delimiter."""
    encapsulated_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("content"),
        delimiters=default_delimiters,
    )
    input_sequence = default_delimiters[0] + "content"
    steppers = encapsulated_acceptor.get_steppers()
    steppers = encapsulated_acceptor.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_rejecting_invalid_content(default_delimiters: tuple[str, str]) -> None:
    """Test rejecting invalid content within the delimiters."""
    encapsulated_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("content"),
        delimiters=default_delimiters,
    )
    input_sequence = default_delimiters[0] + "invalid_content" + default_delimiters[1]
    steppers = encapsulated_acceptor.get_steppers()
    steppers = encapsulated_acceptor.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_accepting_custom_delimiters() -> None:
    """Test handling custom delimiters."""
    custom_open = "<open>"
    custom_close = "</close>"
    encapsulated_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("content"),
        delimiters=(custom_open, custom_close),
    )
    input_sequence = custom_open + "content" + custom_close
    steppers = encapsulated_acceptor.get_steppers()
    steppers = encapsulated_acceptor.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_nested_encapsulated_acceptors():
    """Test nested encapsulated acceptors."""
    inner_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("inner_content"),
        delimiters=("<inner>", "</inner>"),
    )
    outer_acceptor = EncapsulatedStateMachine(
        inner_acceptor,
        delimiters=("<outer>", "</outer>"),
    )
    input_sequence = "<outer><inner>inner_content</inner></outer>"
    steppers = outer_acceptor.get_steppers()
    steppers = outer_acceptor.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_no_acceptance_on_partial_delimiter(default_delimiters: tuple[str, str]) -> None:
    """Test that partial delimiters are not accepted."""
    encapsulated_acceptor = EncapsulatedStateMachine(
        PhraseStateMachine("content"),
        delimiters=default_delimiters,
    )
    partial_open = default_delimiters[0][:-1]
    input_sequence = partial_open + "content" + default_delimiters[1]
    steppers = encapsulated_acceptor.get_steppers()
    steppers = encapsulated_acceptor.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_acceptor_with_whitespace_content(default_delimiters: tuple[str, str]) -> None:
    """Test handling content with whitespace."""
    whitespace_acceptor = PhraseStateMachine("   ")
    encapsulated_acceptor = EncapsulatedStateMachine(
        whitespace_acceptor,
        delimiters=default_delimiters,
    )
    input_sequence = default_delimiters[0] + "   " + default_delimiters[1]
    steppers = encapsulated_acceptor.get_steppers()
    steppers = encapsulated_acceptor.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_accepts_any_token_when_within_value(default_delimiters: tuple[str, str]) -> None:
    """Test that accepts_any_token returns True when within a value."""
    sm = EncapsulatedStateMachine(
        StringStateMachine(),
        delimiters=default_delimiters,
    )
    steppers = sm.get_steppers()
    assert all(stepper.accepts_any_token() for stepper in steppers)

    input_sequence = "Wow, proxy is such a cool company! I would love to work there. Let's see if this unit test passes!"
    steppers = sm.advance_all_basic(steppers, input_sequence)
    assert all(stepper.accepts_any_token() for stepper in steppers)
    assert not any(stepper.is_within_value() for stepper in steppers)
    steppers = sm.advance_all_basic(steppers, default_delimiters[0])

    assert all(not stepper.accepts_any_token() for stepper in steppers)
    assert all(stepper.is_within_value() for stepper in steppers)

    steppers = sm.advance_all_basic(steppers, '"')
    assert all(stepper.is_within_value() for stepper in steppers)
    assert any(stepper.accepts_any_token() for stepper in steppers)


def test_edge_case():
    """Test edge case where the encapsulated acceptor is empty."""
    sm = EncapsulatedStateMachine(
        ObjectStateMachine(),
        delimiters=("<tool>", "</tool>"),
    )
    steppers = sm.get_steppers()
    assert any(stepper.accepts_any_token() for stepper in steppers)
    steppers = sm.advance_all_basic(steppers, '<tool>{"name":"test", "message": {"a":"b"')
    assert len(steppers) == 3

    steppers = sm.advance_all_basic(steppers, "}}")
    steppers = sm.advance_all_basic(steppers, "</tool>")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


def test_min_scratchpad_length():
    """Test that the min_scratchpad_length is respected."""
    sm = EncapsulatedStateMachine(
        StringStateMachine(),
        delimiters=("<tool>", "</tool>"),
        min_buffer_length=10,
    )
    steppers = sm.get_steppers()
    assert not any(stepper.should_start_step("<tool>") for stepper in steppers)
    scratchpad_content = "10+ random characters"
    steppers = sm.advance_all_basic(steppers, scratchpad_content)
    assert any(stepper.should_start_step("<tool>") for stepper in steppers)
    assert not any(stepper.is_within_value() for stepper in steppers)

    steppers = sm.advance_all_basic(steppers, '<tool>"')
    assert all(stepper.is_within_value() for stepper in steppers)
