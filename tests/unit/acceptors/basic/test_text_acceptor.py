import pytest
from pse.state_machine.accepted_state import AcceptedState
from pse.acceptors.basic.text_acceptor import TextAcceptor

@pytest.fixture
def text_acceptor():
    """Fixture to provide a TextAcceptor instance."""
    return TextAcceptor("hello")

def test_advance_incomplete(text_acceptor: TextAcceptor):
    """Test advancing the cursor with an incomplete match."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=0)
    for cursor in cursor.advance("h"):
        assert cursor.get_value() == "h👉ello"
        assert isinstance(cursor, text_acceptor.Cursor)
        assert cursor.consumed_character_count == 1

def test_advance_complete(text_acceptor: TextAcceptor):
    """Test advancing the cursor completes the acceptor."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=4)  # Position before last character
    advanced = list(cursor.advance("o"))
    assert len(advanced) == 1
    accepted = advanced[0]
    assert isinstance(accepted, AcceptedState)
    assert accepted.get_value() == "hello"

def test_advance_invalid_character(text_acceptor: TextAcceptor):
    """Test advancing the cursor with an invalid character does not advance."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=0)
    advanced = list(cursor.advance("x"))
    assert len(advanced) == 0

def test_get_value_at_start(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value at the start."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=0)
    assert cursor.get_value() == "👉hello"

def test_get_value_middle(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value in the middle."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=2)
    assert cursor.get_value() == "he👉llo"

def test_get_value_end(text_acceptor: TextAcceptor):
    """Test the get_value method returns the correct value at the end."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=5)
    assert cursor.get_value() == "hello"

def test_full_acceptance(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor fully accepts the complete string."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=0)
    for char in "hello":
        advanced = list(cursor.advance(char))
        assert len(advanced) == 1
        cursor = advanced[0]
    assert isinstance(cursor, AcceptedState)
    assert cursor.get_value() == "hello"

def test_partial_acceptance(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor correctly handles partial acceptance."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=0)
    for cursor in cursor.advance("he"):
        assert cursor.get_value() == "he👉llo"

def test_repeated_characters():
    """Test the TextAcceptor with repeated characters in the text."""
    repeated_text = "heelloo"
    acceptor = TextAcceptor(repeated_text)
    cursor = acceptor.Cursor(acceptor, consumed_character_count=0)
    for char in repeated_text:
        advanced = list(cursor.advance(char))
        assert len(advanced) == 1
        cursor = advanced[0]
    assert isinstance(cursor, AcceptedState)
    assert cursor.get_value() == repeated_text

def test_unicode_characters():
    """Test the TextAcceptor with Unicode characters."""
    unicode_text = "héllo🌟"
    acceptor = TextAcceptor(unicode_text)
    cursor = acceptor.Cursor(acceptor, consumed_character_count=0)
    for char in unicode_text:
        advanced = list(cursor.advance(char))
        assert len(advanced) == 1
        cursor = advanced[0]
    assert isinstance(cursor, AcceptedState)
    assert cursor.get_value() == unicode_text

def test_empty_text_acceptor():
    """Test that the TextAcceptor raises an assertion error with empty text."""
    with pytest.raises(ValueError):
        TextAcceptor("")

def test_invalid_initial_position(text_acceptor: TextAcceptor):
    """Test that advancing from an invalid initial position does not proceed."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=-1)
    advanced = list(cursor.advance("h"))
    assert len(advanced) == 0

def test_case_sensitivity(text_acceptor: TextAcceptor):
    """Test that the TextAcceptor is case-sensitive."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=0)
    for cursor in cursor.advance("H"):
        assert cursor.get_value() == "👉hello"

    for cursor in cursor.advance("h"):
        assert cursor.get_value() == "h👉ello"

def test_multiple_advance_steps(text_acceptor: TextAcceptor):
    """Test advancing the cursor multiple steps in succession."""
    cursor = text_acceptor.Cursor(text_acceptor, consumed_character_count=0)
    steps = [
        ("h", 1, "h👉ello"),
        ("e", 2, "he👉llo"),
        ("l", 3, "hel👉lo"),
        ("l", 4, "hell👉o"),
        ("o", 5, "hello"),
    ]
    for char, expected_pos, expected_value in steps:
        for cursor in cursor.advance(char):
            assert cursor.get_value() == expected_value
            assert cursor.consumed_character_count == expected_pos

    assert isinstance(cursor, AcceptedState)
    assert cursor.get_value() == "hello"
