from __future__ import annotations

import logging
from typing import List, Type

from pse.core.state_machine import StateMachine, StateMachineWalker
from pse.acceptors.basic.acceptor import Acceptor
from pse.core.walker import Walker

logger = logging.getLogger(__name__)


class SequenceAcceptor(StateMachine):
    """
    Chain multiple TokenAcceptors in a specific sequence.

    Ensures that tokens are accepted in the exact order as defined by the
    sequence of acceptors provided during initialization.
    """

    def __init__(self, acceptors: List[Acceptor]):
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

    @property
    def walker_class(self) -> Type[Walker]:
        return SequenceWalker


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
            and self.current_state not in self.acceptor.end_states
        )
