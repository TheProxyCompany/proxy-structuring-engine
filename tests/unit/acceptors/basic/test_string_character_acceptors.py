import pytest
from pse.acceptors.basic.string_character_acceptor import StringCharacterAcceptor


@pytest.fixture
def string_char_acceptor():
    """Fixture to initialize StringCharacterAcceptor."""
    return StringCharacterAcceptor()

def test_cursor_advance_with_valid_character(string_char_acceptor: StringCharacterAcceptor):
    """
    Test advancing the cursor with a valid character updates the accumulated value.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    initial_value = "he"
    cursor = string_char_acceptor.Cursor(string_char_acceptor, initial_value)
    advance_char = "l"
    advanced_cursors = list(cursor.advance(advance_char))
    assert len(advanced_cursors) == 1, "Should yield one advanced cursor"
    assert advanced_cursors[0].get_value() == "hel", "Accumulated value should be 'hel'"


def test_cursor_advance_with_invalid_character(string_char_acceptor: StringCharacterAcceptor):
    """
    Test advancing the cursor with an invalid character does not update the accumulated value.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    initial_value = "he"
    cursor = string_char_acceptor.Cursor(string_char_acceptor, initial_value)
    advance_char = '"'
    advanced_cursors = list(cursor.advance(advance_char))
    assert (
        len(advanced_cursors) == 0
    ), "Should not yield any advanced cursors for invalid input"

def test_string_char_acceptor_get_value(string_char_acceptor: StringCharacterAcceptor):
    """
    Test that the cursor's get_value method returns the correct accumulated string.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    initial_value = "json"
    cursor = string_char_acceptor.Cursor(string_char_acceptor, initial_value)
    assert (
        cursor.get_value() == initial_value
    ), "get_value should return the correct accumulated string"


def test_string_char_acceptor_cursor_clone(string_char_acceptor: StringCharacterAcceptor):
    """
    Test the cloning functionality of StringCharacterAcceptor.Cursor.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    cursor = string_char_acceptor.Cursor(string_char_acceptor, "clone")
    cloned_cursor = cursor.clone()
    assert (
        cloned_cursor.get_value() == cursor.get_value()
    ), "Cloned cursor should have the same value"
    assert cloned_cursor is not cursor, "Cloned cursor should be a different instance"


def test_string_char_acceptor_empty_cursor_value(string_char_acceptor: StringCharacterAcceptor):
    """
    Test that a cursor with no accumulated value returns None.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    cursor = string_char_acceptor.Cursor(string_char_acceptor)
    assert cursor.get_value() is None, "Cursor with no value should return None"

def test_string_char_acceptor_cursor_is_in_value(string_char_acceptor: StringCharacterAcceptor):
    """
    Test the is_in_value method of StringCharacterAcceptor.Cursor.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    cursor_with_value = string_char_acceptor.Cursor(string_char_acceptor, "value")
    assert (
        cursor_with_value.is_in_value()
    ), "Cursor with value should return True for is_in_value"

    cursor_without_value = string_char_acceptor.Cursor(string_char_acceptor)
    assert (
        not cursor_without_value.is_in_value()
    ), "Cursor without value should return False for is_in_value"
