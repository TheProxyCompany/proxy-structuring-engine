import pytest
from pse_core.accepted_state import AcceptedState
from pse_core.walker import Walker

from pse.state_machines.basic.text_acceptor import TextAcceptor, TextWalker


@pytest.fixture
def test_acceptor() -> TextAcceptor:
    """Fixture to create a TextAcceptor with the text 'test'."""
    return TextAcceptor("test")


@pytest.fixture
def mock_walker(test_acceptor: TextAcceptor) -> Walker:
    """Fixture to create a mock walker from the test_acceptor."""
    return test_acceptor.get_new_walker()


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
        accepted.current_value == mock_walker.current_value
    ), "AcceptedState should return the walker's value."


def test_repr_includes_checkmark_and_walker_repr(
    mock_walker: TextWalker,
) -> None:
    """
    Test that the string representation of AcceptedState includes a checkmark and walker's repr.
    """
    accepted = AcceptedState(mock_walker)
    expected_repr = f"✅ {mock_walker!r}"
    assert (
        repr(accepted) == expected_repr
    ), "String representation should include a checkmark and walker's repr."


def test_multiple_instances_operate_independently() -> None:
    """
    Test that multiple instances of AcceptedState operate independently.
    """
    acceptor1 = TextAcceptor("first")
    walker1 = acceptor1.get_new_walker()
    acceptor2 = TextAcceptor("second")
    walker2 = acceptor2.get_new_walker()
    accepted1 = AcceptedState(walker1)
    accepted2 = AcceptedState(walker2)

    assert accepted1.current_value == "", "First AcceptedState should return ''."
    assert accepted2.current_value == "", "Second AcceptedState should return ''."


def test_equality_of_accepted_state_instances(test_acceptor: TextAcceptor) -> None:
    """
    Test that different AcceptedState instances with identical walker values are not equal.
    """
    walker1 = test_acceptor.get_new_walker()
    walker2 = test_acceptor.get_new_walker()
    accepted1 = AcceptedState(walker1)
    accepted2 = AcceptedState(walker2)

    assert (
        accepted1 != accepted2
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
        accepted.can_accept_more_input() == mock_walker.can_accept_more_input()
    ), "can_handle_remaining_input should match the accepted walker's value."
