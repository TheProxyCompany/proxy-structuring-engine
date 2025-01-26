from __future__ import annotations

from typing import Any

from pse_core import StateGraph, StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.json import JsonStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class ArrayStateMachine(StateMachine):
    """
    Accepts a well-formed JSON array and handles state transitions during parsing.

    This state_machine manages the parsing of JSON arrays by defining the state transitions
    and maintaining the current array values being parsed.
    """

    def __init__(self, state_graph: StateGraph | None = None) -> None:
        base_array_state_graph: StateGraph = {
            0: [(PhraseStateMachine("["), 1)],
            1: [
                (WhitespaceStateMachine(), 2),
                (PhraseStateMachine("]"), "$"),  # Allow empty array
            ],
            2: [(JsonStateMachine(), 3)],
            3: [(WhitespaceStateMachine(), 4)],
            4: [
                (
                    ChainStateMachine(
                        [PhraseStateMachine(","), WhitespaceStateMachine()]
                    ),
                    2,
                ),
                (PhraseStateMachine("]"), "$"),
            ],
        }
        super().__init__(state_graph or base_array_state_graph)

    def get_new_stepper(self, state: StateId | None = None) -> ArrayStepper:
        return ArrayStepper(self, state)

    def __str__(self) -> str:
        return "Array"


class ArrayStepper(Stepper):

    def __init__(
        self,
        state_machine: ArrayStateMachine,
        current_state: StateId | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: ArrayStateMachine = state_machine
        self.value: list[Any] = []

    def clone(self) -> ArrayStepper:
        cloned_stepper = super().clone()
        cloned_stepper.value = self.value[:]
        return cloned_stepper

    def is_within_value(self) -> bool:
        return self.current_state == 3

    def add_to_history(self, stepper: Stepper) -> None:
        if self.is_within_value():
            self.value.append(stepper.get_current_value())
        super().add_to_history(stepper)

    def get_current_value(self) -> list:
        """
        Get the current parsed JSON object.

        Returns:
            dict[str, Any]: The accumulated key-value pairs representing the JSON object.
        """
        if not self.get_raw_value():
            return []
        return self.value
