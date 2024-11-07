import pytest
from typing import Iterable, List

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor, SequenceWalker
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.walker import Walker


@pytest.fixture
def whitespace_acceptor() -> WhitespaceAcceptor:
    """Fixture for WhitespaceAcceptor."""
    return WhitespaceAcceptor()


@pytest.fixture
def hello_acceptor() -> TextAcceptor:
    """Fixture for TextAcceptor with 'Hello'."""
    return TextAcceptor("Hello")


@pytest.fixture
def world_acceptor() -> TextAcceptor:
    """Fixture for TextAcceptor with 'World'."""
    return TextAcceptor("World")


@pytest.fixture
def sequence_acceptor(
    whitespace_acceptor: WhitespaceAcceptor,
    hello_acceptor: TextAcceptor,
    world_acceptor: TextAcceptor,
) -> SequenceAcceptor:
    """Fixture for the default SequenceAcceptor used in tests."""
    return SequenceAcceptor(
        [
            whitespace_acceptor,
            hello_acceptor,
            whitespace_acceptor,
            world_acceptor,
        ]
    )


def test_initialization(sequence_acceptor: SequenceAcceptor):
    """Test that the SequenceAcceptor initializes with the correct number and types of acceptors."""
    expected_acceptor_count = 4
    assert (
        len(sequence_acceptor.acceptors) == expected_acceptor_count
    ), f"SequenceAcceptor should have {expected_acceptor_count} acceptors."

    # Verify each acceptor is of the expected type
    assert isinstance(
        sequence_acceptor.acceptors[0], WhitespaceAcceptor
    ), "First acceptor should be a WhitespaceAcceptor."
    assert isinstance(
        sequence_acceptor.acceptors[1], TextAcceptor
    ), "Second acceptor should be a TextAcceptor for 'Hello'."
    assert isinstance(
        sequence_acceptor.acceptors[2], WhitespaceAcceptor
    ), "Third acceptor should be a WhitespaceAcceptor."
    assert isinstance(
        sequence_acceptor.acceptors[3], TextAcceptor
    ), "Fourth acceptor should be a TextAcceptor for 'World'."


def test_get_walkers(sequence_acceptor: SequenceAcceptor):
    """Test that the SequenceAcceptor initializes with the correct initial walkers."""
    walkers: Iterable[Walker] = sequence_acceptor.get_walkers()
    for walker in walkers:
        assert isinstance(walker, SequenceWalker)


def test_walker_advance(sequence_acceptor: SequenceAcceptor):
    """Test advancing the walker through the sequence of acceptors with specific inputs."""
    start_walkers = list(sequence_acceptor.get_walkers())
    new_walkers = list(sequence_acceptor.advance_all(start_walkers, " "))
    assert len(new_walkers) == 1

    # Advance through the full input sequence " Hello World"
    full_input = " Hello World"
    walkers = sequence_acceptor.get_walkers()
    for char in full_input:
        walkers = StateMachine.advance_all(walkers, char)

    # Verify that at least one walker is in the accepted state
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "The full input ' Hello World' should be accepted by the SequenceAcceptor."


def test_walker_in_accepted_state(sequence_acceptor: SequenceAcceptor):
    """Test the state of the walker before and after processing a complete input sequence."""
    initial_walker = list(sequence_acceptor.get_walkers())[0]
    assert (
        not initial_walker.has_reached_accept_state()
    ), "Initial walker should not be in an accepted state."

    # Process the complete input sequence
    input_sequence = " Hello World"
    walkers = [initial_walker]
    for char in input_sequence:
        walkers = StateMachine.advance_all(walkers, char)

    for walker in walkers:
        assert walker.has_reached_accept_state()
        assert walker.get_current_value() == input_sequence


def test_partial_match(sequence_acceptor: SequenceAcceptor):
    """Test that a partial input sequence does not result in acceptance."""
    partial_input = " Hello "
    walkers = list(sequence_acceptor.get_walkers())
    for char in partial_input:
        walkers = StateMachine.advance_all(walkers, char)

    # Ensure no walker has reached the accepted state with partial input
    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Partial input '{partial_input}' should not be accepted by the SequenceAcceptor."


def test_no_match(sequence_acceptor: SequenceAcceptor):
    """Test that an input sequence not matching the acceptor sequence results in no accepted walkers."""
    non_matching_input = "Goodbye"
    walkers = list(sequence_acceptor.get_walkers())
    for char in non_matching_input:
        walkers = sequence_acceptor.advance_all(walkers, char)

    assert len(list(walkers)) == 0


@pytest.mark.parametrize(
    "input_variant",
    [" Hello World", "\tHello\nWorld", "  Hello  World"],
    ids=["SpaceDelimited", "TabNewlineDelimited", "MultipleSpaces"],
)
def test_whitespace_variations(sequence_acceptor: SequenceAcceptor, input_variant: str):
    """Test that the SequenceAcceptor correctly handles various whitespace variations."""
    walkers = list(sequence_acceptor.get_walkers())
    for char in input_variant:
        walkers = sequence_acceptor.advance_all(walkers, char)
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Input variation '{input_variant}' should be accepted by the SequenceAcceptor."


def test_single_acceptor_sequence():
    """Test that a SequenceAcceptor with a single TextAcceptor correctly accepts the input."""
    single_text = "Test"
    single_acceptor = SequenceAcceptor([TextAcceptor(single_text)])
    walkers = list(single_acceptor.get_walkers())
    for char in single_text:
        walkers = StateMachine.advance_all(walkers, char)
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Single acceptor SequenceAcceptor should accept the input '{single_text}'."


@pytest.mark.parametrize(
    "acceptors, token",
    [
        ([WhitespaceAcceptor(), TextAcceptor("Alpha")], " Alpha"),
        (
            [TextAcceptor("Beta"), WhitespaceAcceptor(), TextAcceptor("Gamma")],
            "Beta Gamma",
        ),
    ],
    ids=["WhitespaceAlphaSequence", "BetaWhitespaceGammaSequence"],
)
def test_multiple_sequences(acceptors: List[TokenAcceptor], token: str):
    """Test multiple SequenceAcceptor instances with different configurations to ensure independence."""
    sequence = SequenceAcceptor(acceptors)
    walkers = list(sequence.get_walkers())
    for char in token:
        walkers = StateMachine.advance_all(walkers, char)
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Input '{token}' should be accepted by the given SequenceAcceptor."
