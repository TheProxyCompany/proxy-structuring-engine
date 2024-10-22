import pytest
from pse.state_machine.cursor import Cursor
from pse.state_machine.accepted_state import AcceptedState
from pse.acceptors.basic.text_acceptor import TextAcceptor


@pytest.fixture
def test_acceptor() -> TextAcceptor:
    """Fixture to create a TextAcceptor with the text 'test'."""
    return TextAcceptor("test")


@pytest.fixture
def mock_cursor(test_acceptor: TextAcceptor) -> TextAcceptor.Cursor:
    """Fixture to create a mock cursor from the test_acceptor."""
    return test_acceptor.Cursor(test_acceptor)


def test_in_accepted_state_returns_true(mock_cursor: TextAcceptor.Cursor) -> None:
    """
    Test that in_accepted_state returns True for AcceptedState.
    """
    accepted = AcceptedState(mock_cursor)
    assert (
        accepted.in_accepted_state()
    ), "AcceptedState should indicate it is in an accepted state."


def test_get_value_returns_correct_value(mock_cursor: TextAcceptor.Cursor) -> None:
    """
    Test that get_value returns the correct value from the cursor.
    """
    accepted = AcceptedState(mock_cursor)
    assert (
        accepted.get_value() == mock_cursor.get_value()
    ), "AcceptedState should return the cursor's value."


def test_repr_includes_checkmark_and_cursor_repr(
    mock_cursor: TextAcceptor.Cursor,
) -> None:
    """
    Test that the string representation of AcceptedState includes a checkmark and cursor's repr.
    """
    accepted = AcceptedState(mock_cursor)
    expected_repr = f"✅{repr(mock_cursor)}"
    assert (
        repr(accepted) == expected_repr
    ), "String representation should include a checkmark and cursor's repr."


def test_multiple_instances_operate_independently() -> None:
    """
    Test that multiple instances of AcceptedState operate independently.
    """
    acceptor1 = TextAcceptor("first")
    cursor1 = acceptor1.Cursor(acceptor1)
    acceptor2 = TextAcceptor("second")
    cursor2 = acceptor2.Cursor(acceptor2)
    accepted1 = AcceptedState(cursor1)
    accepted2 = AcceptedState(cursor2)

    assert (
        accepted1.get_value() == "👉first"
    ), "First AcceptedState should return '👉first'."
    assert (
        accepted2.get_value() == "👉second"
    ), "Second AcceptedState should return '👉second'."
    assert repr(accepted1) != repr(
        accepted2
    ), "String representations should differ based on cursor values."


def test_equality_of_accepted_state_instances(test_acceptor: TextAcceptor) -> None:
    """
    Test that different AcceptedState instances with identical cursor values are not equal.
    """
    cursor1 = test_acceptor.Cursor(test_acceptor)
    cursor2 = test_acceptor.Cursor(test_acceptor)
    accepted1 = AcceptedState(cursor1)
    accepted2 = AcceptedState(cursor2)

    assert (
        accepted1 == accepted2
    ), "AcceptedState instances should not be equal even if their values are the same."

    assert (
        not accepted1.is_in_value()
    ), "AcceptedState should indicate it is not in a value state."
    assert (
        accepted1.in_accepted_state()
    ), "AcceptedState should indicate it is in an accepted state."


def test_advance_delegates_to_accepted_cursor(mock_cursor: TextAcceptor.Cursor) -> None:
    """
    Test that the advance method delegates to the accepted cursor's advance method.
    """
    accepted = AcceptedState(mock_cursor)
    token = "input"
    advances = list(accepted.advance(token))
    expected_advances = list(mock_cursor.advance(token))
    assert (
        advances == expected_advances
    ), "Advance should delegate to the accepted cursor's advance."


def test_can_handle_remaining_input_property(mock_cursor: TextAcceptor.Cursor) -> None:
    """
    Test that can_handle_remaining_input reflects the accepted cursor's ability.
    """
    accepted = AcceptedState(mock_cursor)
    assert (
        accepted.can_handle_remaining_input == mock_cursor.can_handle_remaining_input
    ), "can_handle_remaining_input should match the accepted cursor's value."


def test_get_value_returns_none_when_cursor_value_is_none(
    test_acceptor: TextAcceptor,
) -> None:
    """
    Test that get_value returns None when the cursor's value is None.
    """

    class MockCursor(Cursor):
        """Mock cursor with get_value returning None."""

        remaining_input = ""
        consumed_character_count = 0

        def get_value(self):
            return None

        def in_accepted_state(self):
            return True

        def advance(self, token: str):
            return iter([])

        def is_in_value(self):
            return False

        @property
        def can_handle_remaining_input(self):
            return False

    mock_cursor = MockCursor(test_acceptor)
    accepted = AcceptedState(mock_cursor)
    assert (
        accepted.get_value() is None
    ), "AcceptedState should return None when cursor's get_value is None."
