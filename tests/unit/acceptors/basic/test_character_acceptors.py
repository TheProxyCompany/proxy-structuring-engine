from pse.acceptors.basic.character_acceptors import (
    CharacterAcceptor,
    HexDigitAcceptor,
)


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
    walker = hex_acceptor._walker(hex_acceptor)
    advanced_walkers = list(walker.consume_token("A"))
    for advanced_walker in advanced_walkers:
        assert (
            advanced_walker.raw_value == "A"
        ), "The new walker should have the correct value."


def test_char_acceptor_advance_with_invalid_char():
    """Test advancing CharAcceptor with an invalid character."""
    char_acceptor = CharacterAcceptor(charset="abc")
    walker = char_acceptor._walker(char_acceptor)
    advanced_walkers = list(walker.consume_token("d"))
    for advanced_walker in advanced_walkers:
        assert (
            advanced_walker.raw_value == "d"
        ), "The new walker should have the correct value even if character is invalid."



def test_hex_digit_acceptor_advance_with_lowercase_hex():
    """Test advancing HexDigitAcceptor with a lowercase hexadecimal character."""
    hex_acceptor = HexDigitAcceptor()
    walker = hex_acceptor._walker(hex_acceptor)
    advanced_walkers = list(walker.consume_token("f"))
    for advanced_walker in advanced_walkers:
        assert (
            advanced_walker.raw_value == "f"
        ), "The new walker should have the correct lowercase hex value."
