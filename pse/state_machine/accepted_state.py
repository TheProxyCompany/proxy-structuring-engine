import logging
from typing import Any, Iterable, Optional
from pse.state_machine.cursor import Cursor

logger = logging.getLogger(__name__)

class AcceptedState(Cursor):
    """Represents a cursor that has reached an accepted state.

    This class wraps another cursor (`accepted_cursor`) that has successfully
    reached an accepted state in the state machine. It acts as a marker for
    accepted states and provides methods to retrieve values and advance the cursor.
    """

    def __init__(self, cursor: Cursor) -> None:
        """Initialize the AcceptedState with the given cursor.

        Args:
            cursor: The cursor that has reached an accepted state.
        """
        super().__init__(cursor.acceptor)
        self.accepted_cursor = cursor
        self.current_state = cursor.current_state
        self.remaining_input = cursor.remaining_input
        self.consumed_character_count = cursor.consumed_character_count
        self.accept_history = cursor.accept_history
        self._accepts_remaining_input = cursor._accepts_remaining_input

    @property
    def can_handle_remaining_input(self) -> bool:
        """Determine whether this cursor can handle more input.

        Returns:
            True if the cursor can handle remaining input; False otherwise.
        """
        return self.accepted_cursor.can_handle_remaining_input

    def in_accepted_state(self) -> bool:
        """Check if this cursor is in an accepted state.

        Returns:
            Always `True` for `AcceptedState` instances.
        """
        return True

    def is_in_value(self) -> bool:
        """Determine if this cursor is currently within a value.

        Returns:
            `False`, as accepted states are not considered to be within a value.
        """
        return False

    def get_value(self) -> Any:
        """Retrieve the value from the accepted cursor.

        Returns:
            The value obtained from the accepted cursor.
        """
        return self.accepted_cursor.get_value()

    def advance(self, input_str: str) -> Iterable[Cursor]:
        """Advance the accepted cursor with the given input.

        Args:
            input_str: The input string to process.

        Yields:
            Updated cursors after advancement.
        """
        logger.debug(
            "Advance accepted state %s, can_handle_remaining_input: %s",
            self,
            self.can_handle_remaining_input,
        )

        if not self.can_handle_remaining_input:
            return

        transition_cursor: Optional[Cursor] = None

        if (
            self.accepted_cursor.accept_history
            and self.accepted_cursor.accept_history[-1].can_handle_remaining_input
        ):
            transition_cursor = self.accepted_cursor.accept_history.pop()
            self.accepted_cursor.transition_cursor = transition_cursor
            self.accepted_cursor.target_state = self.accepted_cursor.current_state

        yield from self.accepted_cursor.advance(input_str)

        if transition_cursor:
            self.accepted_cursor.accept_history.append(transition_cursor)

    def is_empty_transition(self) -> bool:
        """Determine if this cursor represents an empty transition.

        Returns:
            `True` if the accepted cursor is an `EmptyTransitionAcceptor.Cursor`; otherwise `False`.
        """
        from .empty_transition import EmptyTransitionAcceptor

        return isinstance(self.accepted_cursor, EmptyTransitionAcceptor.Cursor)

    def __repr__(self) -> str:
        """Return a string representation of the accepted state.

        Returns:
            A string representing the accepted state.
        """
        return f"✅{repr(self.accepted_cursor)}"
