from pse_core import Edge, StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper


class JsonStateMachine(StateMachine):
    def get_edges(self, state: StateId) -> list[Edge]:
        if state == 0:
            from pse.state_machines.base.phrase import PhraseStateMachine
            from pse.state_machines.types.array import ArrayStateMachine
            from pse.state_machines.types.boolean import BooleanStateMachine
            from pse.state_machines.types.number import NumberStateMachine
            from pse.state_machines.types.object import ObjectStateMachine
            from pse.state_machines.types.string import StringStateMachine

            return [
                (ObjectStateMachine(), "$"),
                (ArrayStateMachine(), "$"),
                (StringStateMachine(), "$"),
                (PhraseStateMachine("null"), "$"),
                (BooleanStateMachine(), "$"),
                (NumberStateMachine(), "$"),
            ]
        return []

    def get_steppers(self, state: StateId | None = None) -> list[Stepper]:
        steppers = []
        for edge, _ in self.get_edges(state or 0):
            steppers.extend(edge.get_steppers())
        return steppers
