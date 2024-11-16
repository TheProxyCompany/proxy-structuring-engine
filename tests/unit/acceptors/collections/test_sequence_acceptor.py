import pytest
from typing import List

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.core.state_machine import StateMachine


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


def test_walker_advance(sequence_acceptor: SequenceAcceptor):
    """Test advancing the walker through the sequence of acceptors with specific inputs."""
    start_walkers = list(sequence_acceptor.get_walkers())
    new_walkers = [
        walker
        for _, walker in sequence_acceptor.advance_all_walkers(start_walkers, " ")
    ]
    assert len(new_walkers) == 1

    # Advance through the full input sequence " Hello World"
    full_input = " Hello World"
    walkers = sequence_acceptor.get_walkers()
    for char in full_input:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

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
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    for walker in walkers:
        assert walker.has_reached_accept_state()
        assert walker.current_value == input_sequence


def test_partial_match(sequence_acceptor: SequenceAcceptor):
    """Test that a partial input sequence does not result in acceptance."""
    partial_input = " Hello "
    walkers = list(sequence_acceptor.get_walkers())
    for char in partial_input:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    # Ensure no walker has reached the accepted state with partial input
    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Partial input '{partial_input}' should not be accepted by the SequenceAcceptor."


def test_no_match(sequence_acceptor: SequenceAcceptor):
    """Test that an input sequence not matching the acceptor sequence results in no accepted walkers."""
    non_matching_input = "Goodbye"
    walkers = list(sequence_acceptor.get_walkers())
    for char in non_matching_input:
        walkers = [
            walker for _, walker in sequence_acceptor.advance_all_walkers(walkers, char)
        ]

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
        walkers = [
            walker for _, walker in sequence_acceptor.advance_all_walkers(walkers, char)
        ]
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Input variation '{input_variant}' should be accepted by the SequenceAcceptor."


def test_single_acceptor_sequence():
    """Test that a SequenceAcceptor with a single TextAcceptor correctly accepts the input."""
    single_text = "Test"
    single_acceptor = SequenceAcceptor([TextAcceptor(single_text)])
    walkers = list(single_acceptor.get_walkers())
    for char in single_text:
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]
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
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Input '{token}' should be accepted by the given SequenceAcceptor."


def test_optional_acceptor():
    """Test that an optional acceptor can be used correctly in a SequenceAcceptor."""
    sm = StateMachine(
        graph={
            0: [
                (TextAcceptor("Hello"), 1),
                (TextAcceptor("World"), 1),
            ],
            1: [
                (
                    SequenceAcceptor(
                        [
                            TextAcceptor(","),
                            WhitespaceAcceptor(min_whitespace=0),
                        ]
                    ),
                    0,
                ),
                (TextAcceptor("."), "$"),
            ],
        },
        initial_state=0,
    )
    walkers = list(sm.get_walkers())
    assert len(walkers) == 2
    input_string = "Hello,World."
    new_walkers = [
        walker for _, walker in StateMachine.advance_all_walkers(walkers, input_string)
    ]
    assert len(new_walkers) == 1
    assert new_walkers[0].current_value == input_string
