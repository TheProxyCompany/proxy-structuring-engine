from __future__ import annotations
from typing import List, Optional, Type
from pse.state_machine.state_machine import StateMachine, StateMachineWalker
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.walker import Walker

class SequenceAcceptor(StateMachine):
    """
    Chain multiple TokenAcceptors in a specific sequence.

    Ensures that tokens are accepted in the exact order as defined by the
    sequence of acceptors provided during initialization.
    """

    def __init__(self, acceptors: List[TokenAcceptor], walker_type: Optional[Type[Walker]] = None):
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
        super().__init__(
            graph,
            initial_state=0,
            end_states=[len(acceptors)],
            walker_type=walker_type or SequenceWalker,
        )

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

    def can_accept_more_input(self) -> bool:
        if self.transition_walker and self.transition_walker.can_accept_more_input():
            return True

        return (
            self.current_state not in self.acceptor.end_states
            # and self._accepts_more_input
        )
