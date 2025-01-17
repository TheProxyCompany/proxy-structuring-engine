import pytest
from pse_core.trie import TrieSet

from pse.state_machines.base.phrase import PhraseStateMachine, PhraseWalker


@pytest.fixture
def text_acceptor():
    """Fixture to provide a TextAcceptor instance."""
    return PhraseStateMachine("hello")

def test_basic():
    """Test advancing the walker completes the state_machine."""
    text_acceptor = PhraseStateMachine("hello")
    walker = PhraseWalker(text_acceptor, 4)
    walkers = [walker for _, walker in text_acceptor.advance_all([walker], "o")]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].get_current_value() == "hello"

def test_advance_incomplete():
    """Test advancing the walker completes the state_machine."""
    text_acceptor = PhraseStateMachine("hello")
    walker = PhraseWalker(text_acceptor, 4)
    assert not walker.has_reached_accept_state()
    assert walker.get_current_value() == "hell"


def test_advance_invalid_character(text_acceptor: PhraseStateMachine):
    """Test advancing the walker with an invalid character does not advance."""
    walker = PhraseWalker(text_acceptor, 0)
    advanced = list(walker.consume("x"))
    assert len(advanced) == 0


def test_get_value_at_start(text_acceptor: PhraseStateMachine):
    """Test the get_value method returns the correct value at the start."""
    walker = PhraseWalker(text_acceptor, 0)
    assert walker.get_current_value() is None


def test_get_value_middle(text_acceptor: PhraseStateMachine):
    """Test the get_value method returns the correct value in the middle."""
    walker = PhraseWalker(text_acceptor, 2)
    assert walker.get_current_value() == "he"


def test_get_value_end(text_acceptor: PhraseStateMachine):
    """Test the get_value method returns the correct value at the end."""
    walker = PhraseWalker(text_acceptor, 5)
    assert walker.get_current_value() == "hello"


def test_full_acceptance(text_acceptor: PhraseStateMachine):
    """Test that the TextAcceptor fully accepts the complete string."""
    walker = PhraseWalker(text_acceptor, 0)
    for char in "hello":
        advanced = list(walker.consume(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert walker.has_reached_accept_state()
    assert walker.get_current_value() == "hello"


def test_partial_acceptance(text_acceptor: PhraseStateMachine):
    """Test that the TextAcceptor correctly handles partial acceptance."""
    walker = PhraseWalker(text_acceptor, 0)
    for new_walker in walker.consume("he"):
        assert new_walker.get_current_value() == "he"


def test_repeated_characters():
    """Test the TextAcceptor with repeated characters in the text."""
    repeated_text = "heelloo"
    state_machine = PhraseStateMachine(repeated_text)
    walker = PhraseWalker(state_machine, 0)
    for char in repeated_text:
        advanced = list(walker.consume(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert walker.has_reached_accept_state()
    assert walker.get_current_value() == repeated_text


def test_unicode_characters():
    """Test the TextAcceptor with Unicode characters."""
    unicode_text = "héllo🌟"
    state_machine = PhraseStateMachine(unicode_text)
    walker = PhraseWalker(state_machine, 0)
    for char in unicode_text:
        advanced = walker.consume(char)
        assert len(advanced) == 1
        walker = advanced[0]
    assert walker.has_reached_accept_state()
    assert walker.get_current_value() == unicode_text


def test_empty_text_acceptor():
    """Test that the TextAcceptor raises an assertion error with empty text."""
    with pytest.raises(ValueError):
        PhraseStateMachine("")


def test_invalid_initial_position(text_acceptor: PhraseStateMachine):
    """Test that advancing from an invalid initial position does not proceed."""
    with pytest.raises(ValueError):
        PhraseWalker(text_acceptor, -1)


def test_case_sensitivity(text_acceptor: PhraseStateMachine):
    """Test that the TextAcceptor is case-sensitive."""
    walker = PhraseWalker(text_acceptor, 0)
    for new_walker in walker.consume("H"):
        assert new_walker.get_current_value() == ""

    for new_walker in walker.consume("h"):
        assert new_walker.get_current_value() == "h"


def test_multiple_advance_steps(text_acceptor: PhraseStateMachine):
    """Test advancing the walker multiple steps in succession."""
    walker = PhraseWalker(text_acceptor, 0)
    steps = [
        ("h", 1, "h"),
        ("e", 2, "he"),
        ("l", 3, "hel"),
        ("l", 4, "hell"),
        ("o", 5, "hello"),
    ]
    for char, expected_pos, expected_value in steps:
        for new_walker in walker.consume(char):
            assert new_walker.get_current_value() == expected_value
            assert new_walker.consumed_character_count == expected_pos
            if expected_pos == 5:
                assert new_walker.has_reached_accept_state()


def test_partial_match():
    """Test that the TextAcceptor correctly handles partial matches."""
    text_acceptor = PhraseStateMachine('"hello"')
    walkers = text_acceptor.get_walkers()
    vocab = TrieSet()
    keys = ['"hello', '"', "hello", '"c']
    vocab = vocab.insert_all(keys)
    advanced = text_acceptor.advance_all(walkers, '"*', vocab)
    assert len(advanced) == 1
    for advanced_token, walker in advanced:
        assert walker.get_current_value() == '"'
        assert advanced_token == '"'
