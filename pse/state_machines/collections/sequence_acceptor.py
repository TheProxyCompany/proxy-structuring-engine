from __future__ import annotations

import logging

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

logger = logging.getLogger(__name__)


class SequenceAcceptor(StateMachine):
    """
    Chain multiple TokenAcceptors in a specific sequence.

    Ensures that tokens are accepted in the exact order as defined by the
    sequence of acceptors provided during initialization.
    """

    def __init__(self, acceptors: list[StateMachine]):
        """
        Initialize the SequenceAcceptor with a sequence of TokenAcceptors.

        Args:
            acceptors (list[TokenAcceptor]): An list of TokenAcceptors to be chained.
        """
        # self.acceptors = acceptors
        state_graph = {}
        for i, state_machine in enumerate(acceptors):
            # Each state points **only** to the next state_machine
            state_graph[i] = [(state_machine, i + 1)]
        super().__init__(
            state_graph,
            end_states=[len(acceptors)],
        )

    def get_new_walker(self, state: State | None = None) -> SequenceWalker:
        return SequenceWalker(self, state)


class SequenceWalker(Walker):
    """
    Walker for navigating through the SequenceAcceptor.
    Designed for inspectability and debugging purposes.
    """

    def should_start_transition(self, token: str) -> bool:
        if self.transition_walker and self.transition_walker.state_machine.is_optional:
            return True

        return super().should_start_transition(token)

    def can_accept_more_input(self) -> bool:
        if self.transition_walker and self.transition_walker.can_accept_more_input():
            return True

        return (
            not self.remaining_input
            and self.current_state not in self.state_machine.end_states
        )
