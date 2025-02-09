from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper


class AnyStateMachine(StateMachine):
    def __init__(self, state_machines: list[StateMachine]) -> None:

        self.state_machines: list[StateMachine] = state_machines
        super().__init__(
            {
                0: [
                    (state_machine, "$")
                    for state_machine in self.state_machines
                ]
            }
        )

    def get_steppers(self, state: StateId | None = None) -> list[Stepper]:
        steppers = []
        for edge, _ in self.get_edges(state or 0):
            steppers.extend(edge.get_steppers())
        return steppers

    def __str__(self) -> str:
        return "Any"
