from __future__ import annotations
import pytest
from pse.acceptors.collections.encapsulated_acceptor import EncapsulatedAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor
from pse.util.state_machine.accepted_state import AcceptedState
from pse.core.state_machine import StateMachine


@pytest.fixture
def content_acceptor():
    """Fixture for the content acceptor."""
    return TextAcceptor("content")


@pytest.fixture
def default_delimiters():
    """Fixture for default delimiters."""
    return {
        "open_delimiter": "```json\n",
        "close_delimiter": "\n```",
    }


@pytest.fixture
def default_encapsulated_acceptor(content_acceptor, default_delimiters):
    """Fixture for EncapsulatedAcceptor with default settings."""
    return EncapsulatedAcceptor(
        content_acceptor,
        open_delimiter=default_delimiters["open_delimiter"],
        close_delimiter=default_delimiters["close_delimiter"],
    )


def test_initialization_with_custom_delimiters(content_acceptor):
    """Test initializing EncapsulatedAcceptor with custom delimiters."""
    custom_open = "<start>"
    custom_close = "<end>"
    encapsulated_acceptor = EncapsulatedAcceptor(
        content_acceptor,
        open_delimiter=custom_open,
        close_delimiter=custom_close,
    )

    transitions_state_0 = encapsulated_acceptor.state_graph[0]
    acceptor_0, _ = transitions_state_0[0]
    assert isinstance(acceptor_0, WaitForAcceptor)
    assert isinstance(acceptor_0.wait_for_acceptor, TextAcceptor)
    assert acceptor_0.wait_for_acceptor.text == custom_open

    transitions_state_2 = encapsulated_acceptor.state_graph[2]
    acceptor_2, _ = transitions_state_2[0]
    assert isinstance(acceptor_2, TextAcceptor)
    assert acceptor_2.text == custom_close


def test_accepting_valid_sequence(default_encapsulated_acceptor, default_delimiters):
    """Test accepting a valid sequence with delimiters and content."""
    input_sequence = (
        default_delimiters["open_delimiter"]
        + "content"
        + default_delimiters["close_delimiter"]
    )
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)


def test_rejecting_sequence_missing_open_delimiter(
    default_encapsulated_acceptor, default_delimiters
):
    """Test rejecting sequence missing the open delimiter."""
    input_sequence = "content" + default_delimiters["close_delimiter"]
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_rejecting_sequence_missing_close_delimiter(
    default_encapsulated_acceptor, default_delimiters
):
    """Test rejecting sequence missing the close delimiter."""
    input_sequence = default_delimiters["open_delimiter"] + "content"
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_rejecting_invalid_content(default_encapsulated_acceptor, default_delimiters):
    """Test rejecting invalid content within the delimiters."""
    input_sequence = (
        default_delimiters["open_delimiter"]
        + "invalid_content"
        + default_delimiters["close_delimiter"]
    )
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_accepting_custom_delimiters(content_acceptor):
    """Test handling custom delimiters."""
    custom_open = "<open>"
    custom_close = "</close>"
    encapsulated_acceptor = EncapsulatedAcceptor(
        content_acceptor,
        open_delimiter=custom_open,
        close_delimiter=custom_close,
    )
    input_sequence = custom_open + "content" + custom_close
    walkers = list(encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)


def test_nested_encapsulated_acceptors():
    """Test nested encapsulated acceptors."""
    inner_acceptor = EncapsulatedAcceptor(
        TextAcceptor("inner_content"),
        open_delimiter="<inner>",
        close_delimiter="</inner>",
    )
    outer_acceptor = EncapsulatedAcceptor(
        inner_acceptor, open_delimiter="<outer>", close_delimiter="</outer>"
    )
    input_sequence = "<outer><inner>inner_content</inner></outer>"
    walkers = list(outer_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)


def test_no_acceptance_on_partial_delimiter(
    default_encapsulated_acceptor, default_delimiters
):
    """Test that partial delimiters are not accepted."""
    partial_open = default_delimiters["open_delimiter"][:-1]
    input_sequence = partial_open + "content" + default_delimiters["close_delimiter"]
    walkers = list(default_encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert not any(isinstance(walker, AcceptedState) for walker in walkers)


def test_acceptor_with_empty_content():
    """Test handling of empty content."""
    with pytest.raises(ValueError):
        TextAcceptor("")


def test_acceptor_with_whitespace_content(default_delimiters):
    """Test handling content with whitespace."""
    whitespace_acceptor = TextAcceptor("   ")
    encapsulated_acceptor = EncapsulatedAcceptor(
        whitespace_acceptor,
        open_delimiter=default_delimiters["open_delimiter"],
        close_delimiter=default_delimiters["close_delimiter"],
    )
    input_sequence = (
        default_delimiters["open_delimiter"]
        + "   "
        + default_delimiters["close_delimiter"]
    )
    walkers = list(encapsulated_acceptor.get_walkers())
    for char in input_sequence:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert any(isinstance(walker, AcceptedState) for walker in walkers)
