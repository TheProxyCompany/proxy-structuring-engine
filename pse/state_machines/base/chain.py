from __future__ import annotations

import logging

from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

logger = logging.getLogger(__name__)


class ChainStateMachine(StateMachine):
    """
    Chain multiple StateMachines in a specific order.
    """

    def __init__(self, state_machines: list[StateMachine]) -> None:
        """

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

    def get_new_stepper(self, state: int | str | None = None) -> Stepper:
        return ChainStepper(self, state)

    def __str__(self) -> str:
        return "Chain"


class ChainStepper(Stepper):
    """
    A stepper that chains multiple steppers in a specific sequence.
    """

    def __init__(self, chain_state_machine: ChainStateMachine, *args, **kwargs) -> None:
        super().__init__(chain_state_machine, *args, **kwargs)
        self.state_machine = chain_state_machine
