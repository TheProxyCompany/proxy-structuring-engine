from __future__ import annotations

import logging

from pse_core import State

from pse.state_machine import HierarchicalStateMachine, StateMachineWalker

logger = logging.getLogger(__name__)


class SequenceAcceptor(HierarchicalStateMachine):
    """
    Chain multiple TokenAcceptors in a specific sequence.

    Ensures that tokens are accepted in the exact order as defined by the
    sequence of acceptors provided during initialization.
    """

    def __init__(self, acceptors: list[HierarchicalStateMachine]):
        """
        Initialize the SequenceAcceptor with a sequence of TokenAcceptors.

        Args:
            acceptors (Iterable[TokenAcceptor]): An iterable of TokenAcceptors to be chained.
        """
        self.acceptors = acceptors
        state_graph = {}
        for i, acceptor in enumerate(self.acceptors):
            # Each state points **only** to the next acceptor
            state_graph[i] = [(acceptor, i + 1)]
        super().__init__(
            state_graph,
            end_states=[len(acceptors)],
        )

    def get_new_walker(self, state: State | None = None) -> SequenceWalker:
        return SequenceWalker(self, state)


class SequenceWalker(StateMachineWalker):
    """
    Walker for navigating through the SequenceAcceptor.
    Designed for inspectability and debugging purposes.
    """

    def can_accept_more_input(self) -> bool:
        if self.transition_walker and self.transition_walker.can_accept_more_input():
            return True

        return (
            not self.remaining_input
            and self.current_state not in self.state_machine.end_states
        )
