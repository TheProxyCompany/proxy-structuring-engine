from __future__ import annotations
from typing import Any, List
from pse.state_machine.state_machine import StateMachine
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.cursor import Cursor

class SequenceAcceptor(StateMachine):
    """
    Chain multiple TokenAcceptors in a specific sequence.

    Ensures that tokens are accepted in the exact order as defined by the
    sequence of acceptors provided during initialization.
    """

    def __init__(self, acceptors: List[TokenAcceptor]):
        """
        Initialize the SequenceAcceptor with a sequence of TokenAcceptors.

        Args:
            acceptors (Iterable[TokenAcceptor]): An iterable of TokenAcceptors to be chained.
        """
        self.acceptors = acceptors
        graph = {}
        for i, acceptor in enumerate(self.acceptors):
            # Each state points **only** to the next acceptor
            graph[i] = [(acceptor, i + 1)]
        super().__init__(graph, initial_state=0, end_states=[len(acceptors)])

    def __repr__(self) -> str:
        return f"SequenceAcceptor(acceptors={self.acceptors})"

    def expects_more_input(self, cursor: Cursor) -> bool:
        return cursor.current_state not in self.end_states

    class Cursor(StateMachine.Cursor):
        """
        Cursor for navigating through the SequenceAcceptor.
        Designed for inspectability and debugging purposes.
        """

        def __init__(self, acceptor: SequenceAcceptor, current_acceptor_index: int = 0):
            """
            Initialize the SequenceAcceptor Cursor.

            Args:
                acceptor (SequenceAcceptor): The parent SequenceAcceptor.
                current_acceptor_index (int, optional):
                    The index of the current acceptor in the sequence. Defaults to 0.
            """
            super().__init__(acceptor)
            self.current_acceptor_index: int = current_acceptor_index
            self.acceptor = acceptor

        def get_value(self) -> Any:
            """
            Get the accumulated value from the current acceptor.

            Returns:
                Any: The accumulated value from the current acceptor.
            """
            return "".join([cursor.get_value() for cursor in self.accept_history])

        def __repr__(self) -> str:
            return f"SequenceAcceptor.Cursor(acceptor={self.acceptor}, value={self.get_value()})"
