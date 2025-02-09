from __future__ import annotations

from pse_core import StateGraph, StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.types.base.chain import ChainStateMachine
from pse.types.base.phrase import PhraseStateMachine


class EnumStateMachine(StateMachine):
    """
    Accept one of several constant strings.
    """

    def __init__(self, enum_values: list[str], require_quotes: bool = True) -> None:
        if not enum_values:
            raise ValueError("Enum values must be provided.")

        state_graph: StateGraph = {0: []}
        for value in enum_values:
            sm = (
                PhraseStateMachine(value)
                if not require_quotes
                else ChainStateMachine(
                    [
                        PhraseStateMachine('"'),
                        PhraseStateMachine(value),
                        PhraseStateMachine('"'),
                    ]
                )
            )
            state_graph[0].append((sm, "$"))

        super().__init__(state_graph)

    def get_steppers(self, state: StateId | None = None) -> list[Stepper]:
        steppers = []
        for edge, _ in self.get_edges(state or 0):
            steppers.extend(edge.get_steppers())
        return steppers
