from __future__ import annotations
from typing import Any, Iterable, List
from pse.state_machine.state_machine import StateMachine, StateMachineWalker
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.walker import Walker


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
        """
        Get the string representation of the SequenceAcceptor.
        """
        return f"SequenceAcceptor(acceptors={self.acceptors})"

    def expects_more_input(self, walker: Walker) -> bool:
        """
        Check if the SequenceAcceptor expects more input.
        """
        return walker.current_state not in self.end_states

    def get_walkers(self) -> Iterable[SequenceWalker]:
        """
        Get the walkers for the SequenceAcceptor.
        """
        yield SequenceWalker(self)


class SequenceWalker(StateMachineWalker):
    """
    Walker for navigating through the SequenceAcceptor.
    Designed for inspectability and debugging purposes.
    """

    def __init__(self, acceptor: SequenceAcceptor, current_acceptor_index: int = 0):
        """
        Initialize the SequenceAcceptor Walker.

        Args:
            acceptor (SequenceAcceptor): The parent SequenceAcceptor.
            current_acceptor_index (int, optional):
                The index of the current acceptor in the sequence. Defaults to 0.
        """
        super().__init__(acceptor)
        self.current_acceptor_index: int = current_acceptor_index
        self.acceptor = acceptor

    def accumulated_value(self) -> Any:
        """
        Get the accumulated value from the current acceptor.

        Returns:
            Any: The accumulated value from the current acceptor.
        """
        return "".join([walker.accumulated_value() for walker in self.accept_history])
