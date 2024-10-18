from pse.acceptors.basic.character_acceptors import (
    CharacterAcceptor,
    DigitAcceptor,
    HexDigitAcceptor,
)

def test_digit_acceptor_initialization():
    """Test the initialization of DigitAcceptor."""
    digit_acceptor = DigitAcceptor()
    expected_charset = set("0123456789")
    assert (
        digit_acceptor.charset == expected_charset
    ), "DigitAcceptor should be initialized with digits 0-9."


def test_hex_digit_acceptor_initialization():
    """Test the initialization of HexDigitAcceptor."""
    hex_acceptor = HexDigitAcceptor()
    expected_charset = set("0123456789ABCDEFabcdef")
    assert (
        hex_acceptor.charset == expected_charset
    ), "HexDigitAcceptor should be initialized with hexadecimal digits."


def test_hex_digit_acceptor_cursor_advance():
    """Test the advance method of HexDigitAcceptor.Cursor."""
    hex_acceptor = HexDigitAcceptor()
    cursor = hex_acceptor.Cursor(hex_acceptor)
    advanced_cursors = list(cursor.advance("A"))
    for advanced_cursor in advanced_cursors:
        assert (
            advanced_cursor.get_value() == "A"
        ), "The new cursor should have the correct value."


def test_char_acceptor_advance_with_invalid_char():
    """Test advancing CharAcceptor with an invalid character."""
    char_acceptor = CharacterAcceptor(charset="abc")
    cursor = char_acceptor.Cursor(char_acceptor)
    advanced_cursors = list(cursor.advance("d"))
    for advanced_cursor in advanced_cursors:
        assert (
            advanced_cursor.get_value() == "d"
        ), "The new cursor should have the correct value even if character is invalid."


def test_digit_acceptor_advance_with_valid_digit():
    """Test advancing DigitAcceptor with a valid digit character."""
    digit_acceptor = DigitAcceptor()
    cursor = digit_acceptor.Cursor(digit_acceptor)
    advanced_cursors = list(cursor.advance("5"))
    for advanced_cursor in advanced_cursors:
        assert (
            advanced_cursor.get_value() == "5"
        ), "The new cursor should have the correct digit value."


def test_hex_digit_acceptor_advance_with_lowercase_hex():
    """Test advancing HexDigitAcceptor with a lowercase hexadecimal character."""
    hex_acceptor = HexDigitAcceptor()
    cursor = hex_acceptor.Cursor(hex_acceptor)
    advanced_cursors = list(cursor.advance("f"))
    for advanced_cursor in advanced_cursors:
        assert (
            advanced_cursor.get_value() == "f"
        ), "The new cursor should have the correct lowercase hex value."
