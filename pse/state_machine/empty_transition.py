from typing import Iterator
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.cursor import Cursor


class EmptyTransitionAcceptor(StateMachine):
    """
    Faux acceptor that allows the creation of empty transition edges in a state machine graph.

    This facilitates the expression of complex graphs by skipping the current state without consuming input.
    """

    def get_cursors(self) -> Iterator[Cursor]:
        """
        Retrieve cursors that represent an empty transition.

        Returns:
            Iterator[Cursor]: An iterator containing a single `AcceptedState` cursor.
        """
        from .accepted_state import AcceptedState
        yield AcceptedState(self.Cursor(self))

    class Cursor(StateMachine.Cursor):
        """
        Cursor for handling empty transitions.
        """

        def get_value(self) -> str:
            """
            Retrieve the value associated with the empty transition.

            Returns:
                str: An empty string indicating no consumption.
            """
            return ""

        def __repr__(self) -> str:
            """
            Provide a string representation of the cursor for debugging purposes.

            Returns:
                str: The string representation of the cursor.
            """
            return "EmptyTransition"

        def in_accepted_state(self) -> bool:
            """
            Indicate that this cursor is in an accepted state.

            Returns:
                bool: `True`, as this cursor represents an accepted state.
            """
            return True


EmptyTransition = EmptyTransitionAcceptor({})

__all__ = [
    "EmptyTransition",
]
