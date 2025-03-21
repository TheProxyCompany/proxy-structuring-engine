import pytest
from pse_core.trie import TrieMap

from pse.types.base.phrase import PhraseStateMachine, PhraseStepper


@pytest.fixture
def text_acceptor():
    """Fixture to provide a TextAcceptor instance."""
    return PhraseStateMachine("hello")


def test_basic():
    """Test advancing the stepper completes the state_machine."""
    text_acceptor = PhraseStateMachine("hello")
    stepper = PhraseStepper(text_acceptor, 4)
    steppers = text_acceptor.advance_all_basic([stepper], "o")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == "hello"


def test_advance_incomplete():
    """Test advancing the stepper completes the state_machine."""
    text_acceptor = PhraseStateMachine("hello")
    stepper = PhraseStepper(text_acceptor, 4)
    assert not stepper.has_reached_accept_state()
    assert stepper.get_current_value() == "hell"


def test_advance_invalid_character(text_acceptor: PhraseStateMachine):
    """Test advancing the stepper with an invalid character does not advance."""
    stepper = PhraseStepper(text_acceptor, 0)
    advanced = list(stepper.consume("x"))
    assert len(advanced) == 0


def test_get_value_at_start(text_acceptor: PhraseStateMachine):
    """Test the get_value method returns the correct value at the start."""
    stepper = PhraseStepper(text_acceptor, 0)
    assert stepper.get_current_value() is None


def test_get_value_middle(text_acceptor: PhraseStateMachine):
    """Test the get_value method returns the correct value in the middle."""
    stepper = PhraseStepper(text_acceptor, 2)
    assert stepper.get_current_value() == "he"


def test_get_value_end(text_acceptor: PhraseStateMachine):
    """Test the get_value method returns the correct value at the end."""
    stepper = PhraseStepper(text_acceptor, 5)
    assert stepper.get_current_value() == "hello"


def test_full_acceptance(text_acceptor: PhraseStateMachine):
    """Test that the TextAcceptor fully accepts the complete string."""
    stepper = PhraseStepper(text_acceptor, 0)
    for char in "hello":
        advanced = list(stepper.consume(char))
        assert len(advanced) == 1
        stepper = advanced[0]
    assert stepper.has_reached_accept_state()
    assert stepper.get_current_value() == "hello"


def test_partial_acceptance(text_acceptor: PhraseStateMachine):
    """Test that the TextAcceptor correctly handles partial acceptance."""
    stepper = PhraseStepper(text_acceptor, 0)
    for new_stepper in stepper.consume("he"):
        assert new_stepper.get_current_value() == "he"


def test_repeated_characters():
    """Test the TextAcceptor with repeated characters in the text."""
    repeated_text = "heelloo"
    state_machine = PhraseStateMachine(repeated_text)
    stepper = PhraseStepper(state_machine, 0)
    for char in repeated_text:
        advanced = list(stepper.consume(char))
        assert len(advanced) == 1
        stepper = advanced[0]
    assert stepper.has_reached_accept_state()
    assert stepper.get_current_value() == repeated_text


def test_unicode_characters():
    """Test the TextAcceptor with Unicode characters."""
    unicode_text = "héllo🌟"
    state_machine = PhraseStateMachine(unicode_text)
    stepper = PhraseStepper(state_machine, 0)
    for char in unicode_text:
        advanced = stepper.consume(char)
        assert len(advanced) == 1
        stepper = advanced[0]
    assert stepper.has_reached_accept_state()
    assert stepper.get_current_value() == unicode_text


def test_empty_text_acceptor():
    """Test that the TextAcceptor raises an assertion error with empty text."""
    with pytest.raises(ValueError):
        PhraseStateMachine("")


def test_invalid_initial_position(text_acceptor: PhraseStateMachine):
    """Test that advancing from an invalid initial position does not proceed."""
    with pytest.raises(ValueError):
        PhraseStepper(text_acceptor, -1)


def test_case_sensitivity(text_acceptor: PhraseStateMachine):
    """Test that the TextAcceptor is case-sensitive."""
    stepper = PhraseStepper(text_acceptor, 0)
    for new_stepper in stepper.consume("H"):
        assert new_stepper.get_current_value() == ""

    for new_stepper in stepper.consume("h"):
        assert new_stepper.get_current_value() == "h"


def test_multiple_advance_steps(text_acceptor: PhraseStateMachine):
    """Test advancing the stepper multiple steps in succession."""
    stepper = PhraseStepper(text_acceptor, 0)
    steps = [
        ("h", 1, "h"),
        ("e", 2, "he"),
        ("l", 3, "hel"),
        ("l", 4, "hell"),
        ("o", 5, "hello"),
    ]
    for char, expected_pos, expected_value in steps:
        for new_stepper in stepper.consume(char):
            assert new_stepper.get_current_value() == expected_value
            assert new_stepper.consumed_character_count == expected_pos
            if expected_pos == 5:
                assert new_stepper.has_reached_accept_state()


def test_partial_match():
    """Test that the TextAcceptor correctly handles partial matches."""
    text_acceptor = PhraseStateMachine('"hello"')
    steppers = text_acceptor.get_steppers()
    trie = TrieMap()
    items = [
        ('"hello', 1),
        ('"', 2),
        ("hello", 3),
        ('"c', 4),
    ]
    trie = trie.insert_all(items)
    stepper_deltas = text_acceptor.advance_all(steppers, '"*', vocab=trie)
    assert len(stepper_deltas) == 1
    for stepper_delta in stepper_deltas:
        assert stepper_delta.was_healed
        assert stepper_delta.token == '"'
        assert stepper_delta.stepper.get_current_value() == '"'


def test_get_valid_continuations():
    """Test that the get_valid_continuations returns correct values."""
    text_acceptor = PhraseStateMachine("hello")

    # At the start
    stepper = PhraseStepper(text_acceptor, 0)
    continuations = stepper.get_valid_continuations()
    assert continuations == ["hello"]

    # In the middle
    stepper = PhraseStepper(text_acceptor, 2)
    continuations = stepper.get_valid_continuations()
    assert continuations == ["llo"]

    # At the end
    stepper = PhraseStepper(text_acceptor, 5)
    continuations = stepper.get_valid_continuations()
    assert continuations == []


def test_phrase_stepper_non_equality():
    """Test the equality operator for PhraseStepper."""
    sm1 = PhraseStateMachine("hello")
    sm3 = PhraseStateMachine("world")

    # Same state machine, different position
    stepper1 = PhraseStepper(sm1, 2)
    stepper3 = PhraseStepper(sm1, 3)


    assert stepper1 != stepper3

    # Different state machine
    stepper4 = PhraseStepper(sm3, 2)
    assert stepper1 != stepper4


def test_phrase_state_machine_equality():
    """Test the equality operator for PhraseStateMachine."""
    sm1 = PhraseStateMachine("hello")
    sm2 = PhraseStateMachine("hello")

    assert sm1 == sm2
    stepper1 = PhraseStepper(sm1, 0)
    stepper2 = PhraseStepper(sm2, 0)

    assert stepper1 == stepper2
    stepper1 = sm1.get_new_stepper()
    stepper2 = sm2.get_new_stepper()
    # crashes here
    assert stepper1 == stepper2

def test_should_complete_step():
    """Test the should_complete_step function."""
    text_acceptor = PhraseStateMachine("hello")

    # Not completed
    stepper = PhraseStepper(text_acceptor, 3)
    assert not stepper.should_complete_step()

    # Completed
    stepper = PhraseStepper(text_acceptor, 5)
    assert stepper.should_complete_step()
