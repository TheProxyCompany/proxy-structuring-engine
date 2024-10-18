from __future__ import annotations
from typing import Iterable, Optional, Callable
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.cursor import Cursor


class WaitForAcceptor(StateMachine):
    """
    Accept all text until a segment triggers another specified acceptor.

    This is particularly useful for allowing free-form text until a specific
    delimiter or pattern is detected, such as when parsing output from
    language models that encapsulate JSON within markdown code blocks.
    """

    def __init__(
        self,
        wait_for_acceptor: TokenAcceptor,
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
    ):
        """
        Initialize the WaitForAcceptor with a target acceptor to watch for.

        Args:
            wait_for_acceptor (TokenAcceptor): The acceptor that, when triggered,
                stops the waiting and stops accepting further characters.
        """
        super().__init__({})
        self.wait_for_acceptor = wait_for_acceptor
        self.start_hook = start_hook
        self.end_hook = end_hook
        self.triggered = False

    def get_cursors(self) -> Iterable[Cursor]:
        """
        Retrieve the initial cursor(s) for the WaitForAcceptor.

        Returns:
            Iterable[StateMachineAcceptor.Cursor]: A list containing a single Cursor instance.
        """
        return [WaitForAcceptor.Cursor(self)]

    def advance_cursor(self, cursor: Cursor, input: str) -> Iterable[Cursor]:
        return self.wait_for_acceptor.advance_cursor(cursor, input)

    def __repr__(self) -> str:
        return f"WaitForAcceptor(wait_for_acceptor={self.wait_for_acceptor}, triggered={self.triggered})"

    def expects_more_input(self, cursor: Cursor) -> bool:
        return cursor.current_state not in self.end_states

    class Cursor(Cursor):
        """
        Cursor for handling the WaitForAcceptor.
        Manages internal cursors that monitor for the triggering acceptor.
        """

        def __init__(
            self,
            acceptor: WaitForAcceptor,
            cursors: Optional[Iterable[Cursor]] = None,
        ):
            """
            Initialize the WaitForAcceptor Cursor.

            Args:
                acceptor (WaitForAcceptor): The parent WaitForAcceptor.
                cursors (Optional[Iterable[StateMachineAcceptor.Cursor]], optional):
                    Existing cursors to manage. Defaults to those from the wait_for_acceptor.
            """
            super().__init__(acceptor)
            self.acceptor = acceptor
            if cursors:
                self.cursors = list(cursors)
            else:
                self.cursors = list(self.acceptor.wait_for_acceptor.get_cursors())

        def matches_all(self) -> bool:
            """
            Indicates that this acceptor matches all characters until a trigger is found.

            Returns:
                bool: Always True.
            """
            return True

        def is_in_value(self) -> bool:
            """
            Determine if the cursor is currently within a value.

            Returns:
                bool: True if in a value, False otherwise.
            """
            return any(cursor.is_in_value() for cursor in self.cursors)

        def advance(self, input: str) -> Iterable[Cursor]:
            """
            Advance all internal cursors with the given input.

            Args:
                input (str): The input to process.

            Returns:
                Iterable[TokenAcceptor.Cursor]: Updated cursors after processing.
            """
            new_cursors = []
            for cursor in WaitForAcceptor.advance_all(self.cursors, input):
                if cursor.in_accepted_state():
                    self.acceptor.triggered = True
                    if self.acceptor.end_hook:
                        self.acceptor.end_hook()
                    yield cursor
                    return
                else:
                    new_cursors.append(cursor)

            yield WaitForAcceptor.Cursor(self.acceptor, new_cursors)

        def get_value(self) -> str:
            """
            Retrieve the current value indicating the wait state.

            Returns:
                str: Description of the waiting state.
            """
            return repr(self.acceptor.wait_for_acceptor)

        def __repr__(self) -> str:
            """
            Provide a string representation of the WaitForAcceptor Cursor.

            Returns:
                str: The string representation of the cursor.
            """
            return f"WaitForAcceptor.Cursor(cursors={self.cursors})"
