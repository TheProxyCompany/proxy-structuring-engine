import pytest
from typing import Any, Iterable

from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.walker import Walker
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.types import StateType


class Concretewalker(Walker):
    def __init__(self, acceptor: "TokenAcceptor"):
        super().__init__(acceptor)
        self.value = "test_value"

    def current_value(self) -> Any:
        """Concrete implementation of get_value."""
        return self.value

    def consume_token(self, token: str) -> Iterable[Walker]:
        """Concrete implementation of advance."""
        self.consumed_character_count += len(token)
        self.remaining_input = None
        yield AcceptedState(self)

    def is_within_value(self) -> bool:
        """Concrete implementation of is_in_value."""
        return True

    def has_reached_accept_state(self) -> bool:
        """Concrete implementation of in_accepted_state."""
        return self.current_state in self.acceptor.end_states

    def can_accept_more_input(self) -> bool:
        return False


# Mock TokenAcceptor for testing purposes
class MockTokenAcceptor(TokenAcceptor):
    def __init__(self):
        self.initial_state: StateType = 0
        self.end_states = [1]

    def advance_walker(self, walker: Walker, token: str) -> Iterable[Walker]:
        return super().advance_walker(walker, token)

    def get_walkers(self) -> Iterable[Walker]:
        yield Concretewalker(self)

    def expects_more_input(self, walker: Walker) -> bool:
        return False


@pytest.fixture
def acceptor() -> MockTokenAcceptor:
    return MockTokenAcceptor()


@pytest.fixture
def walker(acceptor: MockTokenAcceptor) -> Concretewalker:
    c = Concretewalker(acceptor)
    return c


def test_walker_initialization(walker: Concretewalker, acceptor: MockTokenAcceptor):
    """Test that the walker initializes correctly."""
    assert walker.acceptor == acceptor
    assert walker.current_state == acceptor.initial_state
    assert walker.target_state is None
    assert walker.transition_walker is None
    assert walker.accepted_history == []
    assert walker.consumed_character_count == 0
    assert walker.remaining_input is None


def test_walker_get_value(walker: Concretewalker):
    """Test the get_value method of Walker."""
    assert walker.current_value() == "test_value"


def test_walker_advance(walker: Concretewalker):
    """Test the advance method of Walker."""
    token = "input"
    walkers = list(walker.consume_token(token))
    assert len(walkers) == 1
    advanced_walker = walkers[0]
    assert advanced_walker.consumed_character_count == len(token)
    assert advanced_walker.remaining_input is None


def test_walker_is_in_value(walker: Concretewalker):
    """Test the is_in_value method of Walker."""
    assert walker.is_within_value() is True


def test_walker_clone(walker: Concretewalker):
    """Test clone method."""
    walker.consumed_character_count = 5
    walker.remaining_input = "remaining"
    clone_walker = walker.clone()
    assert clone_walker is not walker
    assert clone_walker.__dict__ == walker.__dict__


def test_walker_matches_all(walker: Concretewalker):
    """Test matches_all method."""
    assert walker.accepts_any_token() is False


def test_walker_in_accepted_state(walker: Concretewalker):
    """Test in_accepted_state method."""
    walker.current_state = 1  # Set to an end state
    assert walker.has_reached_accept_state() is True


def test_walker_not_in_accepted_state(walker: Concretewalker):
    """Test in_accepted_state method when not in end state."""
    walker.current_state = 0
    assert walker.has_reached_accept_state() is False


def test_walker_equality(walker: Concretewalker):
    """Test __eq__ method."""
    other_walker = walker.clone()
    assert walker == other_walker

    # Change an attribute to make them not equal
    other_walker.current_state = 1
    assert walker != other_walker
