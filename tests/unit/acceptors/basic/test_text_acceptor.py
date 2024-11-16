import pytest
from pse.util.state_machine.accepted_state import AcceptedState
from pse.acceptors.basic.text_acceptor import TextAcceptor, TextWalker


@pytest.fixture
def text_acceptor():
    """Fixture to provide a TextAcceptor instance."""
    return TextAcceptor("hello")


def test_advance_incomplete(text_acceptor: TextAcceptor):
    """Test advancing the walker with an incomplete match."""
    walker = TextWalker(text_acceptor, 0)
    for walker in walker.consume_token("h"):
        assert walker.current_value == "h👉ello"
        assert isinstance(walker, text_acceptor._walker)
        assert walker.consumed_character_count == 1


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
    assert walker.current_value == "👉hello"


def test_get_value_middle(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value in the middle."""
    walker = TextWalker(text_acceptor, 2)
    assert walker.current_value == "he👉llo"


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
    for walker in walker.consume_token("he"):
        assert walker.current_value == "he👉llo"


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
    for walker in walker.consume_token("H"):
        assert walker.current_value == "👉hello"

    for walker in walker.consume_token("h"):
        assert walker.current_value == "h👉ello"


def test_multiple_advance_steps(text_acceptor: TextAcceptor):
    """Test advancing the walker multiple steps in succession."""
    walker = TextWalker(text_acceptor, 0)
    steps = [
        ("h", 1, "h👉ello"),
        ("e", 2, "he👉llo"),
        ("l", 3, "hel👉lo"),
        ("l", 4, "hell👉o"),
        ("o", 5, "hello"),
    ]
    for char, expected_pos, expected_value in steps:
        for walker in walker.consume_token(char):
            assert walker.current_value == expected_value
            assert walker.consumed_character_count == expected_pos

    assert isinstance(walker, AcceptedState)
    assert walker.current_value == "hello"
