import pytest
from pse_core.accepted_state import AcceptedState
from pse_core.trie import TrieSet

from pse.state_machines.basic.text_acceptor import TextAcceptor, TextWalker


def test_basic():
    text_acceptor = TextAcceptor("hello")
    text_walker = TextWalker(text_acceptor, 0)
    text_walker_2 = text_acceptor.get_new_walker()
    assert text_walker == text_walker_2

@pytest.fixture
def text_acceptor():
    """Fixture to provide a TextAcceptor instance."""
    return TextAcceptor("hello")


def test_advance_complete():
    """Test advancing the walker completes the state_machine."""
    text_acceptor = TextAcceptor("hello")
    walker = TextWalker(text_acceptor, 4)
    advanced = list(walker.consume_token("o"))
    assert len(advanced) == 1
    accepted = advanced[0]
    assert isinstance(accepted, AcceptedState)
    assert accepted.get_current_value() == "hello"


def test_advance_invalid_character(text_acceptor: TextAcceptor):
    """Test advancing the walker with an invalid character does not advance."""
    walker = TextWalker(text_acceptor, 0)
    advanced = list(walker.consume_token("x"))
    assert len(advanced) == 0


def test_get_value_at_start(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value at the start."""
    walker = TextWalker(text_acceptor, 0)
    assert walker.get_current_value() == ""


def test_get_value_middle(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value in the middle."""
    walker = TextWalker(text_acceptor, 2)
    assert walker.get_current_value() == "he"


def test_get_value_end(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value at the end."""
    walker = TextWalker(text_acceptor, 5)
    assert walker.get_current_value() == "hello"


def test_full_acceptance(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor fully accepts the complete string."""
    walker = TextWalker(text_acceptor, 0)
    for char in "hello":
        advanced = list(walker.consume_token(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert isinstance(walker, AcceptedState)
    assert walker.get_current_value() == "hello"


def test_partial_acceptance(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor correctly handles partial acceptance."""
    walker = TextWalker(text_acceptor, 0)
    for new_walker in walker.consume_token("he"):
        assert new_walker.get_current_value() == "he"


def test_repeated_characters():
    """Test the TextAcceptor with repeated characters in the text."""
    repeated_text = "heelloo"
    state_machine = TextAcceptor(repeated_text)
    walker = TextWalker(state_machine, 0)
    for char in repeated_text:
        advanced = list(walker.consume_token(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert isinstance(walker, AcceptedState)
    assert walker.get_current_value() == repeated_text


def test_unicode_characters():
    """Test the TextAcceptor with Unicode characters."""
    unicode_text = "héllo🌟"
    state_machine = TextAcceptor(unicode_text)
    walker = TextWalker(state_machine, 0)
    for char in unicode_text:
        advanced = list(walker.consume_token(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert isinstance(walker, AcceptedState)
    assert walker.get_current_value() == unicode_text


def test_empty_text_acceptor():
    """Test that the TextAcceptor raises an assertion error with empty text."""
    with pytest.raises(ValueError):
        TextAcceptor("")


def test_invalid_initial_position(text_acceptor: TextAcceptor):
    """Test that advancing from an invalid initial position does not proceed."""
    with pytest.raises(ValueError):
        TextWalker(text_acceptor, -1)


def test_case_sensitivity(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor is case-sensitive."""
    walker = TextWalker(text_acceptor, 0)
    for new_walker in walker.consume_token("H"):
        assert new_walker.get_current_value() == ""

    for new_walker in walker.consume_token("h"):
        assert new_walker.get_current_value() == "h"


def test_multiple_advance_steps(text_acceptor: TextAcceptor):
    """Test advancing the walker multiple steps in succession."""
    walker = TextWalker(text_acceptor, 0)
    steps = [
        ("h", 1, "h"),
        ("e", 2, "he"),
        ("l", 3, "hel"),
        ("l", 4, "hell"),
        ("o", 5, "hello"),
    ]
    for char, expected_pos, expected_value in steps:
        for new_walker in walker.consume_token(char):
            assert new_walker.get_current_value() == expected_value
            assert new_walker.consumed_character_count == expected_pos
            if expected_pos == 5:
                assert new_walker.has_reached_accept_state()


def test_partial_match():
    """Test that the TextAcceptor correctly handles partial matches."""
    text_acceptor = TextAcceptor('"hello"')
    walkers = text_acceptor.get_walkers()
    vocab = TrieSet()
    keys = ['"hello', '"', "hello", '"c']
    vocab = vocab.insert_all(keys)
    advanced = text_acceptor.advance_all(walkers, '"*', vocab)
    assert len(advanced) == 1
    for advanced_token, walker in advanced:
        assert walker.get_current_value() == '"'
        assert advanced_token == '"'
