import pytest
from pse_core.state_machine import StateMachine

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.string import StringStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


@pytest.fixture(scope="module")
def sequence_acceptor() -> ChainStateMachine:
    """Fixture for the default SequenceAcceptor used in tests."""
    return ChainStateMachine(
        [
            WhitespaceStateMachine(),
            PhraseStateMachine("Hello"),
            WhitespaceStateMachine(),
            PhraseStateMachine("World"),
        ]
    )


def test_basic():
    """Test that an optional state_machine can be used correctly in a SequenceAcceptor."""
    sm = ChainStateMachine(
        [
            PhraseStateMachine(","),
            WhitespaceStateMachine(),
        ]
    )
    walkers = sm.get_walkers()
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, ",")]
    assert len(walkers) == 2

def test_nested_chain():
    sm = StateMachine(
        state_graph={
            0: [
                (
                    ChainStateMachine(
                        [
                            PhraseStateMachine("Hello"),
                            PhraseStateMachine("!", is_optional=True),
                        ]
                    ),
                    1,
                )
            ]
        },
        end_states=[1],
    )
    walkers = sm.get_walkers()
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "Hello")]
    assert len(walkers) == 2
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "!")]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].get_current_value() == "Hello!"


def test_walker_advance(sequence_acceptor: ChainStateMachine):
    """Test advancing the walker through the sequence of acceptors with specific inputs."""
    walkers = sequence_acceptor.get_walkers()
    assert len(walkers) == 2
    walkers = [walker for _, walker in sequence_acceptor.advance_all(walkers, " ")]
    assert len(walkers) == 2
    walkers = [walker for _, walker in sequence_acceptor.advance_all(walkers, "H")]
    assert len(walkers) == 1
    full_input = "ello"
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, full_input)]
    assert len(walkers) == 2
    for walker in walkers:
        assert not walker.has_reached_accept_state()
        assert walker.get_current_value() == " Hello"
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, " World")]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].get_current_value() == " Hello World"

    # Verify that at least one walker is in the accepted state
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "The full input ' Hello World' should be accepted by the SequenceAcceptor."


def test_walker_in_accepted_state(sequence_acceptor: ChainStateMachine):
    """Test the state of the walker before and after processing a complete input sequence."""
    initial_walker = next(iter(sequence_acceptor.get_walkers()))
    assert (
        not initial_walker.has_reached_accept_state()
    ), "Initial walker should not be in an accepted state."

    # Process the complete input sequence
    input_sequence = " Hello World"
    walkers = [initial_walker]
    for char in input_sequence:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    for walker in walkers:
        assert walker.has_reached_accept_state()
        assert walker.get_current_value() == input_sequence


def test_partial_match(sequence_acceptor: ChainStateMachine):
    """Test that a partial input sequence does not result in acceptance."""
    partial_input = " Hello "
    walkers = list(sequence_acceptor.get_walkers())
    for char in partial_input:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    # Ensure no walker has reached the accepted state with partial input
    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Partial input '{partial_input}' should not be accepted by the SequenceAcceptor."


def test_no_match(sequence_acceptor: ChainStateMachine):
    """Test that an input sequence not matching the state_machine sequence results in no accepted walkers."""
    non_matching_input = "Goodbye"
    walkers = list(sequence_acceptor.get_walkers())
    for char in non_matching_input:
        walkers = [walker for _, walker in sequence_acceptor.advance_all(walkers, char)]

    assert len(list(walkers)) == 0


@pytest.mark.parametrize(
    "input_variant",
    [" Hello World", "\tHello\nWorld", "  Hello  World"],
    ids=["SpaceDelimited", "TabNewlineDelimited", "MultipleSpaces"],
)
def test_whitespace_variations(
    sequence_acceptor: ChainStateMachine, input_variant: str
):
    """Test that the SequenceAcceptor correctly handles various whitespace variations."""
    walkers = list(sequence_acceptor.get_walkers())
    for char in input_variant:
        walkers = [walker for _, walker in sequence_acceptor.advance_all(walkers, char)]
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Input variation '{input_variant}' should be accepted by the SequenceAcceptor."


def test_single_acceptor_sequence():
    """Test that a SequenceAcceptor with a single TextAcceptor correctly accepts the input."""
    single_text = "Test"
    single_acceptor = ChainStateMachine([PhraseStateMachine(single_text)])
    walkers = list(single_acceptor.get_walkers())
    for char in single_text:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]
    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Single state_machine SequenceAcceptor should accept the input '{single_text}'."


def test_whitespace_first():
    """Test that the WhitespaceAcceptor is first in the sequence."""
    sequence = ChainStateMachine(
        [WhitespaceStateMachine(), PhraseStateMachine(" Alpha")]
    )
    walkers = sequence.get_walkers()
    assert len(walkers) == 2
    for char in " Alpha":
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_whitespace_middle():
    """Test that the WhitespaceAcceptor is in the middle of the sequence."""
    sequence = ChainStateMachine(
        [
            PhraseStateMachine("Beta"),
            WhitespaceStateMachine(),
            PhraseStateMachine("Gamma"),
        ]
    )
    walkers = sequence.get_walkers()
    for char in "Beta Gamma":
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_optional_acceptor_advanced():
    """Test that an optional state_machine can be used correctly in a SequenceAcceptor."""
    sm = StateMachine(
        state_graph={
            0: [
                (PhraseStateMachine("Hello"), 1),
                (PhraseStateMachine("World"), 1),
            ],
            1: [
                (
                    ChainStateMachine(
                        [
                            PhraseStateMachine(","),
                            WhitespaceStateMachine(min_whitespace=0),
                        ]
                    ),
                    0,
                ),
                (PhraseStateMachine("."), "$"),
            ],
        },
        start_state=0,
    )
    walkers = sm.get_walkers()
    assert len(walkers) == 2
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "Hello")]
    assert len(walkers) == 2
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, ",")]
    assert len(walkers) == 3
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, " ")]
    assert len(walkers) == 3
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "World.")]
    assert len(walkers) == 1
    assert walkers[0].get_current_value() == "Hello, World."
    assert walkers[0].has_reached_accept_state()
    walkers = sm.get_walkers()
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "Hello,")]
    assert len(walkers) == 3
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "World.")]
    assert len(walkers) == 1
    assert walkers[0].get_current_value() == "Hello,World."
    assert walkers[0].has_reached_accept_state()


def test_whitespace_acceptor_sequence_acceptor():
    """Test WhitespaceAcceptor in the context of SequenceAcceptor."""
    sequence_acceptor = ChainStateMachine(
        [
            StringStateMachine(),
            WhitespaceStateMachine(),
            PhraseStateMachine(":"),
            WhitespaceStateMachine(),
        ]
    )
    walkers = sequence_acceptor.get_walkers()
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, '"')]
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, 'test"')]
    assert len(walkers) == 2
    for walker in walkers:
        assert not walker.has_reached_accept_state()
        assert walker.get_raw_value() == '"test"'

    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "   ")]
    assert len(walkers) == 2
    for walker in walkers:
        assert not walker.has_reached_accept_state()
        assert walker.get_raw_value() == '"test"   '

    walkers = [walker for _, walker in StateMachine.advance_all(walkers, ":    ")]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].get_current_value() == '"test"   :    '
