import pytest

from pse.acceptors.basic.string_character_acceptor import (
    StringCharacterAcceptor,
    StringCharacterWalker,
)


@pytest.fixture
def string_char_acceptor():
    """Fixture to initialize StringCharacterAcceptor."""
    return StringCharacterAcceptor()


def test_walker_advance_with_valid_character(
    string_char_acceptor: StringCharacterAcceptor,
):
    """
    Test advancing the walker with a valid character updates the accumulated value.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    initial_value = "he"
    walker = StringCharacterWalker(string_char_acceptor, initial_value)
    advance_char = "l"
    advanced_walkers = list(walker.consume_token(advance_char))
    assert len(advanced_walkers) == 1, "Should yield one advanced walker"
    assert (
        advanced_walkers[0].current_value == "hel"
    ), "Accumulated value should be 'hel'"


def test_walker_advance_with_invalid_character(
    string_char_acceptor: StringCharacterAcceptor,
):
    """
    Test advancing the walker with an invalid character does not update the accumulated value.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    initial_value = "he"
    walker = StringCharacterWalker(string_char_acceptor, initial_value)
    advance_char = '"'
    advanced_walkers = list(walker.consume_token(advance_char))
    assert (
        len(advanced_walkers) == 0
    ), "Should not yield any advanced walkers for invalid input"


def test_string_char_acceptor_get_value(string_char_acceptor: StringCharacterAcceptor):
    """
    Test that the walker's get_value method returns the correct accumulated string.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    initial_value = "json"
    walker = StringCharacterWalker(string_char_acceptor, initial_value)
    assert (
        walker.current_value == initial_value
    ), "get_value should return the correct accumulated string"


def test_string_char_acceptor_walker_clone(
    string_char_acceptor: StringCharacterAcceptor,
):
    """
    Test the cloning functionality of StringCharacterAcceptor.Walker.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    walker = StringCharacterWalker(string_char_acceptor, "clone")
    cloned_walker = walker.clone()
    assert (
        cloned_walker.current_value == walker.current_value
    ), "Cloned walker should have the same value"
    assert cloned_walker is not walker, "Cloned walker should be a different instance"


def test_string_char_acceptor_empty_walker_value(
    string_char_acceptor: StringCharacterAcceptor,
):
    """
    Test that a walker with no accumulated value returns None.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    walker = StringCharacterWalker(string_char_acceptor)
    assert walker.current_value is None, "Walker with no value should return None"


def test_string_char_acceptor_walker_is_in_value(
    string_char_acceptor: StringCharacterAcceptor,
):
    """
    Test the is_in_value method of StringCharacterAcceptor.Walker.

    Args:
        string_char_acceptor (StringCharacterAcceptor): The acceptor instance.
    """
    walker_with_value = StringCharacterWalker(string_char_acceptor, "value")
    assert (
        walker_with_value.is_within_value()
    ), "Walker with value should return True for is_in_value"

    walker_without_value = StringCharacterWalker(string_char_acceptor)
    assert (
        not walker_without_value.is_within_value()
    ), "Walker without value should return False for is_in_value"
