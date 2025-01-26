from __future__ import annotations

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.base.phrase import PhraseStateMachine


class BooleanStateMachine(StateMachine):
    """
    Accepts a JSON boolean value: true, false.
    """

    def __init__(self) -> None:
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("true"), "$"),
                    (PhraseStateMachine("false"), "$"),
                ]
            }
        )

    def get_steppers(self, state: StateId | None = None) -> list[Stepper]:
        steppers = []
        for edge, _ in self.get_edges(state or 0):
            steppers.extend(edge.get_steppers())
        return steppers
