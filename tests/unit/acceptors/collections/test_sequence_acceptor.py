import pytest
from typing import Iterable, List

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.cursor import Cursor


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
    assert len(sequence_acceptor.acceptors) == expected_acceptor_count, (
        f"SequenceAcceptor should have {expected_acceptor_count} acceptors."
    )

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


def test_get_cursors(sequence_acceptor: SequenceAcceptor):
    """Test that the SequenceAcceptor initializes with the correct initial cursors."""
    cursors: Iterable[Cursor] = sequence_acceptor.get_cursors()
    for cursor in cursors:
        assert isinstance(cursor, SequenceAcceptor.Cursor)


def test_cursor_advance(sequence_acceptor: SequenceAcceptor):
    """Test advancing the cursor through the sequence of acceptors with specific inputs."""
    start_cursors = list(sequence_acceptor.get_cursors())
    new_cursors = list(sequence_acceptor.advance_all(start_cursors, " "))
    assert len(new_cursors) == 1

    # Advance through the full input sequence " Hello World"
    full_input = " Hello World"
    cursors = sequence_acceptor.get_cursors()
    for char in full_input:
        cursors = StateMachine.advance_all(cursors, char)

    # Verify that at least one cursor is in the accepted state
    assert any(
        cursor.in_accepted_state() for cursor in cursors
    ), "The full input ' Hello World' should be accepted by the SequenceAcceptor."


def test_cursor_in_accepted_state(sequence_acceptor: SequenceAcceptor):
    """Test the state of the cursor before and after processing a complete input sequence."""
    initial_cursor = list(sequence_acceptor.get_cursors())[0]
    assert (
        not initial_cursor.in_accepted_state()
    ), "Initial cursor should not be in an accepted state."

    # Process the complete input sequence
    input_sequence = " Hello World"
    cursors = [initial_cursor]
    for char in input_sequence:
        cursors = StateMachine.advance_all(cursors, char)

    for cursor in cursors:
        assert cursor.in_accepted_state()
        assert cursor.get_value() == input_sequence


def test_partial_match(sequence_acceptor: SequenceAcceptor):
    """Test that a partial input sequence does not result in acceptance."""
    partial_input = " Hello "
    cursors = list(sequence_acceptor.get_cursors())
    for char in partial_input:
        cursors = StateMachine.advance_all(cursors, char)

    # Ensure no cursor has reached the accepted state with partial input
    assert not any(
        cursor.in_accepted_state() for cursor in cursors
    ), f"Partial input '{partial_input}' should not be accepted by the SequenceAcceptor."


def test_no_match(sequence_acceptor: SequenceAcceptor):
    """Test that an input sequence not matching the acceptor sequence results in no accepted cursors."""
    non_matching_input = "Goodbye"
    cursors = list(sequence_acceptor.get_cursors())
    for char in non_matching_input:
        cursors = sequence_acceptor.advance_all(cursors, char)

    assert len(list(cursors)) == 0


@pytest.mark.parametrize(
    "input_variant",
    [" Hello World", "\tHello\nWorld", "  Hello  World"],
    ids=["SpaceDelimited", "TabNewlineDelimited", "MultipleSpaces"],
)
def test_whitespace_variations(sequence_acceptor: SequenceAcceptor, input_variant: str):
    """Test that the SequenceAcceptor correctly handles various whitespace variations."""
    cursors = list(sequence_acceptor.get_cursors())
    for char in input_variant:
        cursors = sequence_acceptor.advance_all(cursors, char)
    assert any(
        cursor.in_accepted_state() for cursor in cursors
    ), f"Input variation '{input_variant}' should be accepted by the SequenceAcceptor."


def test_single_acceptor_sequence():
    """Test that a SequenceAcceptor with a single TextAcceptor correctly accepts the input."""
    single_text = "Test"
    single_acceptor = SequenceAcceptor([TextAcceptor(single_text)])
    cursors = list(single_acceptor.get_cursors())
    for char in single_text:
        cursors = StateMachine.advance_all(cursors, char)
    assert any(
        cursor.in_accepted_state() for cursor in cursors
    ), f"Single acceptor SequenceAcceptor should accept the input '{single_text}'."


@pytest.mark.parametrize(
    "acceptors, input_str",
    [
        ([WhitespaceAcceptor(), TextAcceptor("Alpha")], " Alpha"),
        (
            [TextAcceptor("Beta"), WhitespaceAcceptor(), TextAcceptor("Gamma")],
            "Beta Gamma",
        ),
    ],
    ids=["WhitespaceAlphaSequence", "BetaWhitespaceGammaSequence"],
)
def test_multiple_sequences(acceptors: List[TokenAcceptor], input_str: str):
    """Test multiple SequenceAcceptor instances with different configurations to ensure independence."""
    sequence = SequenceAcceptor(acceptors)
    cursors = list(sequence.get_cursors())
    for char in input_str:
        cursors = StateMachine.advance_all(cursors, char)
    assert any(
        cursor.in_accepted_state() for cursor in cursors
    ), f"Input '{input_str}' should be accepted by the given SequenceAcceptor."
