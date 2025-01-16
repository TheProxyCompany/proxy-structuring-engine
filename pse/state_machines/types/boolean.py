from __future__ import annotations

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine


class BooleanStateMachine(StateMachine):
    """
    Accepts a JSON boolean value: true, false.
    """

    def __init__(self) -> None:
        """
        Initialize the BooleanAcceptor with its state transitions defined as a state graph.
        """
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("true"), "$"),
                    (PhraseStateMachine("false"), "$"),
                ]
            }
        )

    def get_walkers(self, state: State | None = None) -> list[Walker]:
        walkers = []
        for edge, _ in self.get_edges(state or 0):
            walkers.extend(edge.get_walkers())
        return walkers
