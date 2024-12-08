import pytest
from pse_core.accepted_state import AcceptedState

from pse.acceptors.basic.text_acceptor import TextAcceptor, TextWalker


@pytest.fixture
def text_acceptor():
    """Fixture to provide a TextAcceptor instance."""
    return TextAcceptor("hello")


def test_advance_complete(text_acceptor: TextAcceptor):
    """Test advancing the walker completes the acceptor."""
    walker = TextWalker(text_acceptor, 4)  # Position before last character
    advanced = list(walker.consume_token("o"))
    assert len(advanced) == 1
    accepted = advanced[0]
    assert isinstance(accepted, AcceptedState)
    assert accepted.current_value == "hello"


def test_advance_invalid_character(text_acceptor: TextAcceptor):
    """Test advancing the walker with an invalid character does not advance."""
    walker = TextWalker(text_acceptor, 0)
    advanced = list(walker.consume_token("x"))
    assert len(advanced) == 0


def test_get_value_at_start(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value at the start."""
    walker = TextWalker(text_acceptor, 0)
    assert walker.current_value == ""


def test_get_value_middle(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value in the middle."""
    walker = TextWalker(text_acceptor, 2)
    assert walker.current_value == "he"


def test_get_value_end(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value at the end."""
    walker = TextWalker(text_acceptor, 5)
    assert walker.current_value == "hello"


def test_full_acceptance(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor fully accepts the complete string."""
    walker = TextWalker(text_acceptor, 0)
    for char in "hello":
        advanced = list(walker.consume_token(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert isinstance(walker, AcceptedState)
    assert walker.current_value == "hello"


def test_partial_acceptance(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor correctly handles partial acceptance."""
    walker = TextWalker(text_acceptor, 0)
    for new_walker in walker.consume_token("he"):
        assert new_walker.current_value == "he"


def test_repeated_characters():
    """Test the TextAcceptor with repeated characters in the text."""
    repeated_text = "heelloo"
    acceptor = TextAcceptor(repeated_text)
    walker = TextWalker(acceptor, 0)
    for char in repeated_text:
        advanced = list(walker.consume_token(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert isinstance(walker, AcceptedState)
    assert walker.current_value == repeated_text


def test_unicode_characters():
    """Test the TextAcceptor with Unicode characters."""
    unicode_text = "héllo🌟"
    acceptor = TextAcceptor(unicode_text)
    walker = TextWalker(acceptor, 0)
    for char in unicode_text:
        advanced = list(walker.consume_token(char))
        assert len(advanced) == 1
        walker = advanced[0]
    assert isinstance(walker, AcceptedState)
    assert walker.current_value == unicode_text


def test_empty_text_acceptor():
    """Test that the TextAcceptor raises an assertion error with empty text."""
    with pytest.raises(ValueError):
        TextAcceptor("")


def test_invalid_initial_position(text_acceptor: TextAcceptor):
    """Test that advancing from an invalid initial position does not proceed."""
    walker = TextWalker(text_acceptor, -1)
    advanced = list(walker.consume_token("h"))
    assert len(advanced) == 0


def test_case_sensitivity(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor is case-sensitive."""
    walker = TextWalker(text_acceptor, 0)
    for new_walker in walker.consume_token("H"):
        assert new_walker.current_value == ""

    for new_walker in walker.consume_token("h"):
        assert new_walker.current_value == "h"


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
            assert new_walker.current_value == expected_value
            assert new_walker.consumed_character_count == expected_pos

    assert isinstance(walker, AcceptedState)
    assert walker.current_value == "hello"
