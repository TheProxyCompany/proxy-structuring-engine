from __future__ import annotations

import logging
from typing import Any

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.key_value import KeyValueStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine

logger = logging.getLogger()


class ObjectStateMachine(StateMachine):
    """
    Accepts a well-formed JSON object and manages state transitions during parsing.

    This state_machine handles the parsing of JSON objects by defining state transitions
    and maintaining the current object properties being parsed.
    """

    def __init__(self, is_optional: bool = False) -> None:
        """

        Sets up the state transition graph for parsing JSON objects.
        """
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("{"), 1),
                ],
                1: [
                    (WhitespaceStateMachine(), 2),
                ],
                2: [
                    (KeyValueStateMachine(), 3),
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
                    (PhraseStateMachine("}"), "$"),  # End of object
                ],
            },
            is_optional=is_optional,
        )

    def get_new_stepper(self, state: StateId | None = None) -> ObjectStepper:
        return ObjectStepper(self, state)

    def get_transitions(self, stepper: Stepper) -> list[tuple[Stepper, StateId]]:
        transitions = super().get_transitions(stepper)
        if stepper.current_state == 1 and self.is_optional:
            for transition in PhraseStateMachine("}").get_steppers():
                transitions.append((transition, "$"))
        return transitions

    def __str__(self) -> str:
        return "Object"


class ObjectStepper(Stepper):
    def __init__(
        self, state_machine: ObjectStateMachine, current_state: StateId | None = None
    ) -> None:
        super().__init__(state_machine, current_state)
        self.value: dict[str, Any] = {}

    def clone(self) -> ObjectStepper:
        cloned_stepper = super().clone()
        cloned_stepper.value = self.value.copy()
        return cloned_stepper

    def is_within_value(self) -> bool:
        return self.consumed_character_count > 0

    def add_to_history(self, stepper: Stepper) -> None:
        if self.current_state == 3:
            prop_name, prop_value = stepper.get_current_value()
            logger.debug(f"🟢 Adding {prop_name}: {prop_value} to {self.value}")
            self.value[prop_name] = prop_value
        super().add_to_history(stepper)

    def get_current_value(self) -> dict[str, Any]:
        """
        Get the current parsed JSON object.

        Returns:
            dict[str, Any]: The accumulated key-value pairs representing the JSON object.
        """
        if not self.get_raw_value():
            return {}
        return self.value
