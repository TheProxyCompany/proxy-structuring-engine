from __future__ import annotations

import logging

from pse_core.state_machine import StateMachine

logger = logging.getLogger(__name__)


class ChainStateMachine(StateMachine):
    """
    Chain multiple StateMachines in a specific sequence.

    Ensures that tokens are accepted in the exact order as defined by the
    sequence of acceptors provided during initialization.
    """

    def __init__(self, state_machines: list[StateMachine]) -> None:
        """
        Initialize the SequenceAcceptor with a sequence of state machines.

        Args:
            state_machines (list[StateMachine]): A list of state machines to be chained in sequence.
        """
        super().__init__(
            state_graph={
                i: [(state_machine, i + 1)]
                for i, state_machine in enumerate(state_machines)
            },
            end_states=[len(state_machines)],
        )

    def __str__(self) -> str:
        return "Chain"
