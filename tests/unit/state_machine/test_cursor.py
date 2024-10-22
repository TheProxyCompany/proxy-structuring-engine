import pytest
from typing import Any, Iterable, Type

from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.cursor import Cursor
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.types import StateType


class ConcreteCursor(Cursor):
    def __init__(self, acceptor: "TokenAcceptor"):
        super().__init__(acceptor)
        self.value = "test_value"

    def get_value(self) -> Any:
        """Concrete implementation of get_value."""
        return self.value

    def advance(self, token: str) -> Iterable[Cursor]:
        """Concrete implementation of advance."""
        self.consumed_character_count += len(token)
        self.remaining_input = None
        yield AcceptedState(self)

    def is_in_value(self) -> bool:
        """Concrete implementation of is_in_value."""
        return True

    def in_accepted_state(self) -> bool:
        """Concrete implementation of in_accepted_state."""
        return self.current_state in self.acceptor.end_states


# Mock TokenAcceptor for testing purposes
class MockTokenAcceptor(TokenAcceptor):
    def __init__(self):
        self.initial_state: StateType = 0
        self.end_states = [1]

    def advance_cursor(self, cursor: Cursor, token: str) -> Iterable[Cursor]:
        return super().advance_cursor(cursor, token)

    @property
    def cursor_class(self) -> Type[Cursor]:
        return ConcreteCursor

    def expects_more_input(self, cursor: Cursor) -> bool:
        return False


@pytest.fixture
def acceptor() -> MockTokenAcceptor:
    return MockTokenAcceptor()


@pytest.fixture
def cursor(acceptor: MockTokenAcceptor) -> ConcreteCursor:
    c = ConcreteCursor(acceptor)
    return c


def test_cursor_initialization(cursor: ConcreteCursor, acceptor: MockTokenAcceptor):
    """Test that the cursor initializes correctly."""
    assert cursor.acceptor == acceptor
    assert cursor.current_state == acceptor.initial_state
    assert cursor.target_state is None
    assert cursor.transition_cursor is None
    assert cursor.accept_history == []
    assert cursor.consumed_character_count == 0
    assert cursor.remaining_input is None


def test_cursor_get_value(cursor: ConcreteCursor):
    """Test the get_value method of Cursor."""
    assert cursor.get_value() == "test_value"


def test_cursor_advance(cursor: ConcreteCursor):
    """Test the advance method of Cursor."""
    token = "input"
    cursors = list(cursor.advance(token))
    assert len(cursors) == 1
    advanced_cursor = cursors[0]
    assert advanced_cursor.consumed_character_count == len(token)
    assert advanced_cursor.remaining_input is None


def test_cursor_is_in_value(cursor: ConcreteCursor):
    """Test the is_in_value method of Cursor."""
    assert cursor.is_in_value() is True


def test_cursor_start_transition(cursor: ConcreteCursor, acceptor: MockTokenAcceptor):
    """Test start_transition method."""
    result = cursor.start_transition(acceptor, 1)
    assert result is True


def test_cursor_complete_transition(cursor: ConcreteCursor):
    """Test complete_transition method."""
    result = cursor.complete_transition(None, 1, False)
    assert result is True


def test_cursor_clone(cursor: ConcreteCursor):
    """Test clone method."""
    cursor.consumed_character_count = 5
    cursor.remaining_input = "remaining"
    clone_cursor = cursor.clone()
    assert clone_cursor is not cursor
    assert clone_cursor.__dict__ == cursor.__dict__


def test_cursor_matches_all(cursor: ConcreteCursor):
    """Test matches_all method."""
    assert cursor.matches_all() is False


def test_cursor_in_accepted_state(cursor: ConcreteCursor):
    """Test in_accepted_state method."""
    cursor.current_state = 1  # Set to an end state
    assert cursor.in_accepted_state() is True


def test_cursor_not_in_accepted_state(cursor: ConcreteCursor):
    """Test in_accepted_state method when not in end state."""
    cursor.current_state = 0
    assert cursor.in_accepted_state() is False


def test_cursor_equality(cursor: ConcreteCursor):
    """Test __eq__ method."""
    other_cursor = cursor.clone()
    assert cursor == other_cursor

    # Change an attribute to make them not equal
    other_cursor.current_state = 1
    assert cursor != other_cursor
