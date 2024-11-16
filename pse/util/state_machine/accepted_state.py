import logging
from typing import Any, Iterable
from pse.core.walker import Walker

logger = logging.getLogger(__name__)


class AcceptedState(Walker):
    """Represents a walker that has reached an accepted state.

    This class wraps another walker (`accepted_walker`) that has successfully
    reached an accepted state in the state machine. It acts as a marker for
    accepted states and provides methods to retrieve values and advance the walker.
    """

    def __init__(self, walker: Walker) -> None:
        """Initialize the AcceptedState with the given walker.

        Args:
            walker: The walker that has reached an accepted state.
        """
        self.acceptor = walker.acceptor
        self.accepted_walker = walker
        self.accepted_history = walker.accepted_history
        self.explored_edges = walker.explored_edges

        self.current_state = walker.current_state
        self.target_state = walker.target_state
        self.transition_walker = walker.transition_walker

        self.remaining_input = walker.remaining_input
        self.consumed_character_count = walker.consumed_character_count

        self._raw_value = walker.raw_value
        self._accepts_more_input = walker._accepts_more_input

    def clone(self) -> Walker:
        return self.accepted_walker.clone()

    def can_accept_more_input(self) -> bool:
        """Check if the accepted walker can accept more input.

        Returns:
            True if the accepted walker can accept more input; False otherwise.
        """
        return self.accepted_walker.can_accept_more_input()

    def has_reached_accept_state(self) -> bool:
        """Check if this walker is in an accepted state.

        Returns:
            Always `True` for `AcceptedState` instances.
        """
        return True

    def is_within_value(self) -> bool:
        """Determine if this walker is currently within a value.

        Returns:
            `False`, as accepted states are not considered to be within a value.
        """
        return False

    @property
    def current_value(self) -> Any:
        """Retrieve the value from the accepted walker.

        Returns:
            The value obtained from the accepted walker.
        """
        return self.accepted_walker.current_value

    def should_start_transition(self, token: str) -> bool:
        """Determines if a transition should start with the given input string.

        Args:
            token: The input string to process.

        Returns:
            True if the transition should start; False otherwise.
        """
        if not self.can_accept_more_input():
            return False

        return self.accepted_walker.should_start_transition(token)

    def consume_token(self, token: str) -> Iterable[Walker]:
        """Advance the accepted walker with the given input.

        Args:
            token: The input string to process.

        Yields:
            Updated walkers after advancement.
        """
        if not self.can_accept_more_input():
            return

        yield from self.accepted_walker.consume_token(token)

    def __eq__(self, other: Any) -> bool:
        return self.accepted_walker.__eq__(other)

    def __hash__(self) -> int:
        return self.accepted_walker.__hash__()

    def __repr__(self) -> str:
        """Return a string representation of the accepted state.

        Returns:
            A string representing the accepted state.
        """
        return f"✅ {repr(self.accepted_walker)}"
