import pytest
from pse.state_machine.walker import Walker
from pse.state_machine.accepted_state import AcceptedState
from pse.acceptors.basic.text_acceptor import TextAcceptor, TextWalker


@pytest.fixture
def test_acceptor() -> TextAcceptor:
    """Fixture to create a TextAcceptor with the text 'test'."""
    return TextAcceptor("test")


@pytest.fixture
def mock_walker(test_acceptor: TextAcceptor) -> TextWalker:
    """Fixture to create a mock walker from the test_acceptor."""
    return test_acceptor.walker_class(test_acceptor)


def test_in_accepted_state_returns_true(mock_walker: TextWalker) -> None:
    """
    Test that in_accepted_state returns True for AcceptedState.
    """
    accepted = AcceptedState(mock_walker)
    assert (
        accepted.has_reached_accept_state()
    ), "AcceptedState should indicate it is in an accepted state."


def test_get_value_returns_correct_value(mock_walker: TextWalker) -> None:
    """
    Test that get_value returns the correct value from the walker.
    """
    accepted = AcceptedState(mock_walker)
    assert (
        accepted.accumulated_value() == mock_walker.accumulated_value()
    ), "AcceptedState should return the walker's value."


def test_repr_includes_checkmark_and_walker_repr(
    mock_walker: TextWalker,
) -> None:
    """
    Test that the string representation of AcceptedState includes a checkmark and walker's repr.
    """
    accepted = AcceptedState(mock_walker)
    expected_repr = f"✅{repr(mock_walker)}"
    assert (
        repr(accepted) == expected_repr
    ), "String representation should include a checkmark and walker's repr."


def test_multiple_instances_operate_independently() -> None:
    """
    Test that multiple instances of AcceptedState operate independently.
    """
    acceptor1 = TextAcceptor("first")
    walker1 = acceptor1.walker_class(acceptor1)
    acceptor2 = TextAcceptor("second")
    walker2 = acceptor2.walker_class(acceptor2)
    accepted1 = AcceptedState(walker1)
    accepted2 = AcceptedState(walker2)

    assert (
        accepted1.accumulated_value() == "👉first"
    ), "First AcceptedState should return '👉first'."
    assert (
        accepted2.accumulated_value() == "👉second"
    ), "Second AcceptedState should return '👉second'."
    assert repr(accepted1) != repr(
        accepted2
    ), "String representations should differ based on walker values."


def test_equality_of_accepted_state_instances(test_acceptor: TextAcceptor) -> None:
    """
    Test that different AcceptedState instances with identical walker values are not equal.
    """
    walker1 = test_acceptor.walker_class(test_acceptor)
    walker2 = test_acceptor.walker_class(test_acceptor)
    accepted1 = AcceptedState(walker1)
    accepted2 = AcceptedState(walker2)

    assert (
        accepted1 == accepted2
    ), "AcceptedState instances should not be equal even if their values are the same."

    assert (
        not accepted1.is_within_value()
    ), "AcceptedState should indicate it is not in a value state."
    assert (
        accepted1.has_reached_accept_state()
    ), "AcceptedState should indicate it is in an accepted state."


def test_advance_delegates_to_accepted_walker(mock_walker: TextWalker) -> None:
    """
    Test that the advance method delegates to the accepted walker's advance method.
    """
    accepted = AcceptedState(mock_walker)
    token = "input"
    advances = list(accepted.consume_token(token))
    expected_advances = list(mock_walker.consume_token(token))
    assert (
        advances == expected_advances
    ), "Advance should delegate to the accepted walker's advance."


def test_can_handle_remaining_input_property(mock_walker: TextWalker) -> None:
    """
    Test that can_handle_remaining_input reflects the accepted walker's ability.
    """
    accepted = AcceptedState(mock_walker)
    assert (
        accepted.can_handle_remaining_input == mock_walker.can_handle_remaining_input
    ), "can_handle_remaining_input should match the accepted walker's value."


def test_get_value_returns_none_when_walker_value_is_none(
    test_acceptor: TextAcceptor,
) -> None:
    """
    Test that get_value returns None when the walker's value is None.
    """

    class Mockwalker(Walker):
        """Mock walker with get_value returning None."""

        remaining_input = ""
        consumed_character_count = 0

        def accumulated_value(self):
            return None

        def has_reached_accept_state(self):
            return True

        def consume_token(self, token: str):
            return iter([])

        def is_within_value(self):
            return False

        @property
        def can_handle_remaining_input(self):
            return False

    mock_walker = Mockwalker(test_acceptor)
    accepted = AcceptedState(mock_walker)
    assert (
        accepted.accumulated_value() is None
    ), "AcceptedState should return None when walker's get_value is None."
