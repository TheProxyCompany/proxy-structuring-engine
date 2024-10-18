from __future__ import annotations

from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor
from pse.state_machine.cursor import Cursor
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
)


class EncapsulatedAcceptor(StateMachine):
    """
    Accepts JSON data within a larger text, delimited by specific markers.

    This class encapsulates an acceptor that recognizes JSON content framed by
    specified opening and closing delimiters.
    """

    def __init__(
        self,
        acceptor: TokenAcceptor,
        open_delimiter: str,
        close_delimiter: str,
    ) -> None:
        """
        Initialize the EncapsulatedAcceptor with delimiters and the JSON acceptor.

        Args:
            acceptor: The acceptor responsible for validating the JSON content.
            open_delimiter: The string that denotes the start of the JSON content.
            close_delimiter: The string that denotes the end of the JSON content.
        """
        graph: StateMachineGraph = {
            0: [
                (WaitForAcceptor(TextAcceptor(open_delimiter)), 1),
            ],
            1: [
                (acceptor, 2),
            ],
            2: [(TextAcceptor(close_delimiter), "$")],
        }
        self.opening_delimiter: str = open_delimiter
        self.closing_delimiter: str = close_delimiter
        self.wait_for_acceptor: TokenAcceptor = acceptor
        super().__init__(graph)

    def expects_more_input(self, cursor: Cursor) -> bool:
        return cursor.current_state not in self.end_states

    # -------- Nested Classes --------

    class Cursor(StateMachine.Cursor):
        """
        Cursor for the EncapsulatedAcceptor.
        """

        def __init__(self, acceptor: EncapsulatedAcceptor) -> None:
            """
            Initialize the Cursor for EncapsulatedAcceptor.

            Args:
                acceptor: The EncapsulatedAcceptor instance this cursor belongs to.
            """
            super().__init__(acceptor)
            self.acceptor: EncapsulatedAcceptor = acceptor

        def is_in_value(self) -> bool:
            """
            Determine if the cursor is currently within a value.

            Returns:
                bool: True if in a value, False otherwise.
            """
            return (
                self.current_state != self.acceptor.initial_state
                or (self.transition_cursor is not None and self.transition_cursor.is_in_value())
            )
