from __future__ import annotations

import pytest
from pse_core.accepted_state import AcceptedState
from pse_core.state_machine import StateMachine

from pse.state_machines.basic.string_acceptor import StringAcceptor
from pse.state_machines.basic.text_acceptor import TextAcceptor
from pse.state_machines.collections.encapsulated_acceptor import EncapsulatedAcceptor
from pse.state_machines.collections.wait_for_acceptor import WaitForAcceptor


@pytest.fixture
def content_acceptor():
    """Fixture for the content state_machine."""
    return TextAcceptor("content")


@pytest.fixture
def default_delimiters() -> tuple[str, str]:
    """Fixture for default delimiters."""
    return "```json\n", "\n```"


@pytest.fixture
def default_encapsulated_acceptor(content_acceptor, default_delimiters) -> EncapsulatedAcceptor:
    """Fixture for EncapsulatedAcceptor with default settings."""
    return EncapsulatedAcceptor(
        content_acceptor,
        delimiters=default_delimiters,
    )


def test_initialization_with_custom_delimiters(content_acceptor):
    """Test initializing EncapsulatedAcceptor with custom delimiters."""
    custom_open = "<start>"
    custom_close = "<end>"
    encapsulated_acceptor = EncapsulatedAcceptor(
        content_acceptor,
        delimiters=(custom_open, custom_close),
    )

    transitions_state_0 = encapsulated_acceptor.state_graph[0]
    acceptor_0, _ = transitions_state_0[0]
    assert isinstance(acceptor_0, WaitForAcceptor)
    assert isinstance(acceptor_0.wait_for_sm, TextAcceptor)
    assert acceptor_0.wait_for_sm.text == custom_open

    transitions_state_2 = encapsulated_acceptor.state_graph[2]
    acceptor_2, _ = transitions_state_2[0]
    assert isinstance(acceptor_2, TextAcceptor)
    assert acceptor_2.text == custom_close


def test_accepting_valid_sequence(
    default_encapsulated_acceptor: EncapsulatedAcceptor,
    default_delimiters: tuple[str, str],
) -> None:
    """Test accepting a valid sequence with delimiters and content."""
    input_sequence = (
        default_delimiters[0]
        + "content"
        + default_delimiters[1]
    )
    walkers = default_encapsulated_acceptor.get_walkers()
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)


def test_rejecting_sequence_missing_open_delimiter(
    default_encapsulated_acceptor, default_delimiters
):
    """Test rejecting sequence missing the open delimiter."""
    input_sequence = "content" + default_delimiters[1]
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_rejecting_sequence_missing_close_delimiter(
    default_encapsulated_acceptor, default_delimiters
):
    """Test rejecting sequence missing the close delimiter."""
    input_sequence = default_delimiters[0] + "content"
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_rejecting_invalid_content(default_encapsulated_acceptor, default_delimiters):
    """Test rejecting invalid content within the delimiters."""
    input_sequence = (
        default_delimiters[0]
        + "invalid_content"
        + default_delimiters[1]
    )
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_accepting_custom_delimiters(content_acceptor):
    """Test handling custom delimiters."""
    custom_open = "<open>"
    custom_close = "</close>"
    encapsulated_acceptor = EncapsulatedAcceptor(
        content_acceptor,
        delimiters=(custom_open, custom_close),
    )
    input_sequence = custom_open + "content" + custom_close
    walkers = encapsulated_acceptor.get_walkers()
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)


def test_nested_encapsulated_acceptors():
    """Test nested encapsulated acceptors."""
    inner_acceptor = EncapsulatedAcceptor(
        TextAcceptor("inner_content"),
        delimiters=("<inner>", "</inner>"),
    )
    outer_acceptor = EncapsulatedAcceptor(
        inner_acceptor,
        delimiters=("<outer>", "</outer>"),
    )
    input_sequence = "<outer><inner>inner_content</inner></outer>"
    walkers = outer_acceptor.get_walkers()
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)


def test_no_acceptance_on_partial_delimiter(
    default_encapsulated_acceptor, default_delimiters
):
    """Test that partial delimiters are not accepted."""
    partial_open = default_delimiters[0][:-1]
    input_sequence = partial_open + "content" + default_delimiters[1]
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_acceptor_with_whitespace_content(default_delimiters):
    """Test handling content with whitespace."""
    whitespace_acceptor = TextAcceptor("   ")
    encapsulated_acceptor = EncapsulatedAcceptor(
        whitespace_acceptor,
        delimiters=default_delimiters,
    )
    input_sequence = (
        default_delimiters[0]
        + "   "
        + default_delimiters[1]
    )
    walkers = encapsulated_acceptor.get_walkers()
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)


def test_accepts_any_token_when_within_value(default_delimiters):
    """Test that accepts_any_token returns True when within a value."""
    sm = EncapsulatedAcceptor(
        StringAcceptor(),
        delimiters=default_delimiters,
    )
    walkers = sm.get_walkers()
    assert all(walker.accepts_any_token() for walker in walkers)

    input_sequence = "Wow, proxy is such a cool company! I would love to work there. Let's see if this unit test passes!"
    walkers = [
        walker
        for _, walker in StateMachine.advance_all(
            walkers, input_sequence
        )
    ]
    assert all(walker.accepts_any_token() for walker in walkers)
    assert not any(walker.is_within_value() for walker in walkers)

    walkers = [walker for _, walker in StateMachine.advance_all(walkers, default_delimiters[0])]

    assert all(not walker.accepts_any_token() for walker in walkers)
    assert all(walker.is_within_value() for walker in walkers)

    walkers = [walker for _, walker in StateMachine.advance_all(walkers, '"')]
    assert all(walker.is_within_value() for walker in walkers)
    assert any(walker.accepts_any_token() for walker in walkers)
