from __future__ import annotations

from typing import Any

from pse_core import StateId
from pse_core.stepper import Stepper

from pse.state_machines import schema_to_state_machine
from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.array import ArrayStateMachine, ArrayStepper
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class ArraySchemaStateMachine(ArrayStateMachine):
    def __init__(self, schema: dict[str, Any], context: dict[str, Any]) -> None:
        self.schema = schema
        self.context = context
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("["), 1),
                ],
                1: [
                    (WhitespaceStateMachine(), 2),
                    (PhraseStateMachine("]"), "$"),
                ],
                2: [
                    (schema_to_state_machine(self.schema["items"], self.context), 3),
                ],
                3: [
                    (WhitespaceStateMachine(), 4),
                ],
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
        )

    def get_transitions(self, stepper: Stepper) -> list[tuple[Stepper, StateId]]:
        """Retrieve transition steppers from the current state.

        For each edge from the current state, returns steppers that can traverse that edge.
        Args:
            stepper: The stepper initiating the transition.
            state: Optional starting state. If None, uses the stepper's current state.

        Returns:
            list[tuple[Stepper, StateId]]: A list of tuples representing transitions.
        """
        if stepper.current_state == 4:
            transitions: list[tuple[Stepper, StateId]] = []
            if len(stepper.get_current_value()) >= self.min_items():
                for transition in PhraseStateMachine("]").get_steppers():
                    transitions.append((transition, "$"))

            if len(stepper.get_current_value()) < self.max_items():
                for transition in ChainStateMachine(
                    [PhraseStateMachine(","), WhitespaceStateMachine()]
                ).get_steppers():
                    transitions.append((transition, 2))

            return transitions
        elif stepper.current_state == 1 and self.min_items() > 0:
            transitions = []
            for transition in WhitespaceStateMachine().get_steppers():
                transitions.append((transition, 2))
            return transitions
        else:
            return super().get_transitions(stepper)

    def get_new_stepper(self, state: StateId | None = None) -> ArraySchemaStepper:
        return ArraySchemaStepper(self, state)

    def min_items(self) -> int:
        """
        Returns the minimum number of items in the array, according to the schema
        """
        return self.schema.get("minItems", 0)

    def max_items(self) -> int:
        """
        Returns the maximum number of items in the array, according to the schema
        """
        return self.schema.get("maxItems", 2**32)

    def unique_items(self) -> bool:
        """
        Returns whether the items in the array must be unique, according to the schema
        """
        return self.schema.get("uniqueItems", False)

    def __str__(self) -> str:
        return super().__str__() + "Schema"


class ArraySchemaStepper(ArrayStepper):
    """ """

    def __init__(
        self,
        state_machine: ArraySchemaStateMachine,
        current_state: StateId | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: ArraySchemaStateMachine = state_machine

    def add_to_history(self, stepper: Stepper) -> None:
        """
        Adds an item to the array.
        """
        item = stepper.get_current_value()
        if self.state_machine.unique_items() and self.is_within_value():
            if item in self.value:
                return

        super().add_to_history(stepper)
