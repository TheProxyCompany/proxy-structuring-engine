import pytest
from pse_core.state_machine import StateMachine

from pse.base.chain import ChainStateMachine
from pse.base.phrase import PhraseStateMachine
from pse.types.string import StringStateMachine
from pse.types.whitespace import WhitespaceStateMachine


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
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, ",")
    assert len(steppers) == 2


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
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "Hello")
    assert len(steppers) == 2
    steppers = sm.advance_all_basic(steppers, "!")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == "Hello!"


def test_stepper_advance(sequence_acceptor: ChainStateMachine):
    """
    Test advancing the stepper through the sequence of acceptors with specific inputs.

    This is a more complex test that ensures the stepper can handle various inputs and
    whitespace variations.
    """
    steppers = sequence_acceptor.get_steppers()
    assert len(steppers) == 2
    steppers = sequence_acceptor.advance_all_basic(steppers, " ")
    assert len(steppers) == 2
    steppers = sequence_acceptor.advance_all_basic(steppers, "H")
    assert len(steppers) == 1
    full_input = "ello"
    steppers = sequence_acceptor.advance_all_basic(steppers, full_input)
    assert len(steppers) == 2
    for stepper in steppers:
        assert not stepper.has_reached_accept_state()
        assert stepper.get_current_value() == " Hello"
    steppers = sequence_acceptor.advance_all_basic(steppers, " World")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == " Hello World"

    # Verify that at least one stepper is in the accepted state
    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "The full input ' Hello World' should be accepted by the SequenceAcceptor."
    )


def test_stepper_in_accepted_state(sequence_acceptor: ChainStateMachine):
    """Test the state of the stepper before and after processing a complete input sequence."""
    initial_stepper = next(iter(sequence_acceptor.get_steppers()))
    assert not initial_stepper.has_reached_accept_state(), (
        "Initial stepper should not be in an accepted state."
    )

    # Process the complete input sequence
    input_sequence = " Hello World"
    steppers = [initial_stepper]
    for char in input_sequence:
        steppers = sequence_acceptor.advance_all_basic(steppers, char)

    for stepper in steppers:
        assert stepper.has_reached_accept_state()
        assert stepper.get_current_value() == input_sequence


def test_partial_match(sequence_acceptor: ChainStateMachine):
    """Test that a partial input sequence does not result in acceptance."""
    partial_input = " Hello "
    steppers = list(sequence_acceptor.get_steppers())
    for char in partial_input:
        steppers = sequence_acceptor.advance_all_basic(steppers, char)

    # Ensure no stepper has reached the accepted state with partial input
    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"Partial input '{partial_input}' should not be accepted by the SequenceAcceptor."
    )


def test_no_match(sequence_acceptor: ChainStateMachine):
    """Test that an input sequence not matching the state_machine sequence results in no accepted steppers."""
    non_matching_input = "Goodbye"
    steppers = list(sequence_acceptor.get_steppers())
    for char in non_matching_input:
        steppers = sequence_acceptor.advance_all_basic(steppers, char)

    assert len(list(steppers)) == 0


@pytest.mark.parametrize(
    "input_variant",
    [" Hello World", "\tHello\nWorld", "  Hello  World"],
    ids=["SpaceDelimited", "TabNewlineDelimited", "MultipleSpaces"],
)
def test_whitespace_variations(
    sequence_acceptor: ChainStateMachine, input_variant: str
):
    """Test that the SequenceAcceptor correctly handles various whitespace variations."""
    steppers = list(sequence_acceptor.get_steppers())
    for char in input_variant:
        steppers = sequence_acceptor.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"Input variation '{input_variant}' should be accepted by the SequenceAcceptor."
    )


def test_single_acceptor_sequence():
    """Test that a SequenceAcceptor with a single TextAcceptor correctly accepts the input."""
    single_text = "Test"
    single_acceptor = ChainStateMachine([PhraseStateMachine(single_text)])
    steppers = list(single_acceptor.get_steppers())
    for char in single_text:
        steppers = single_acceptor.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"Single state_machine SequenceAcceptor should accept the input '{single_text}'."
    )


def test_whitespace_first():
    """Test that the WhitespaceAcceptor is first in the sequence."""
    sequence = ChainStateMachine(
        [WhitespaceStateMachine(), PhraseStateMachine(" Alpha")]
    )
    steppers = sequence.get_steppers()
    assert len(steppers) == 2
    for char in " Alpha":
        steppers = sequence.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_whitespace_middle():
    """Test that the WhitespaceAcceptor is in the middle of the sequence."""
    sequence = ChainStateMachine(
        [
            PhraseStateMachine("Beta"),
            WhitespaceStateMachine(),
            PhraseStateMachine("Gamma"),
        ]
    )
    steppers = sequence.get_steppers()
    for char in "Beta Gamma":
        steppers = sequence.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


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
    steppers = sm.get_steppers()
    assert len(steppers) == 2
    steppers = sm.advance_all_basic(steppers, "Hello")
    assert len(steppers) == 2
    steppers = sm.advance_all_basic(steppers, ",")
    assert len(steppers) == 3
    steppers = sm.advance_all_basic(steppers, " ")
    assert len(steppers) == 3
    steppers = sm.advance_all_basic(steppers, "World.")
    assert len(steppers) == 1
    assert steppers[0].get_current_value() == "Hello, World."
    assert steppers[0].has_reached_accept_state()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "Hello,")
    assert len(steppers) == 3
    steppers = sm.advance_all_basic(steppers, "World.")
    assert len(steppers) == 1
    assert steppers[0].get_current_value() == "Hello,World."
    assert steppers[0].has_reached_accept_state()


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
    steppers = sequence_acceptor.get_steppers()
    steppers = sequence_acceptor.advance_all_basic(steppers, '"')
    steppers = sequence_acceptor.advance_all_basic(steppers, 'test"')
    assert len(steppers) == 2
    for stepper in steppers:
        assert not stepper.has_reached_accept_state()
        assert stepper.get_raw_value() == '"test"'

    steppers = sequence_acceptor.advance_all_basic(steppers, "   ")
    assert len(steppers) == 2
    for stepper in steppers:
        assert not stepper.has_reached_accept_state()
        assert stepper.get_raw_value() == '"test"   '

    steppers = sequence_acceptor.advance_all_basic(steppers, ":    ")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == '"test"   :    '
