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


def test_hex_digit_acceptor_walker_advance():
    """Test the advance method of HexDigitAcceptor.Walker."""
    hex_acceptor = HexDigitAcceptor()
    walker = hex_acceptor.walker_class(hex_acceptor)
    advanced_walkers = list(walker.consume_token("A"))
    for advanced_walker in advanced_walkers:
        assert (
            advanced_walker.accumulated_value() == "A"
        ), "The new walker should have the correct value."


def test_char_acceptor_advance_with_invalid_char():
    """Test advancing CharAcceptor with an invalid character."""
    char_acceptor = CharacterAcceptor(charset="abc")
    walker = char_acceptor.walker_class(char_acceptor)
    advanced_walkers = list(walker.consume_token("d"))
    for advanced_walker in advanced_walkers:
        assert (
            advanced_walker.accumulated_value() == "d"
        ), "The new walker should have the correct value even if character is invalid."


def test_digit_acceptor_advance_with_valid_digit():
    """Test advancing DigitAcceptor with a valid digit character."""
    digit_acceptor = DigitAcceptor()
    walker = digit_acceptor.walker_class(digit_acceptor)
    advanced_walkers = list(walker.consume_token("5"))
    for advanced_walker in advanced_walkers:
        assert (
            advanced_walker.accumulated_value() == "5"
        ), "The new walker should have the correct digit value."


def test_hex_digit_acceptor_advance_with_lowercase_hex():
    """Test advancing HexDigitAcceptor with a lowercase hexadecimal character."""
    hex_acceptor = HexDigitAcceptor()
    walker = hex_acceptor.walker_class(hex_acceptor)
    advanced_walkers = list(walker.consume_token("f"))
    for advanced_walker in advanced_walkers:
        assert (
            advanced_walker.accumulated_value() == "f"
        ), "The new walker should have the correct lowercase hex value."
