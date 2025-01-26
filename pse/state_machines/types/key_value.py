from __future__ import annotations

import json
import logging
from typing import Any

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.json import JsonStateMachine
from pse.state_machines.types.string import StringStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine

logger = logging.getLogger()


class KeyValueStateMachine(ChainStateMachine):
    def __init__(self, sequence: list[StateMachine] | None = None) -> None:
        super().__init__(
            sequence
            or [
                StringStateMachine(),
                WhitespaceStateMachine(),
                PhraseStateMachine(":"),
                WhitespaceStateMachine(),
                JsonStateMachine(),
            ]
        )

    def get_new_stepper(self, state: StateId | None = None) -> KeyValueStepper:
        return KeyValueStepper(self, state)

    def __str__(self) -> str:
        return "KeyValue"


class KeyValueStepper(Stepper):
    def __init__(
        self,
        state_machine: KeyValueStateMachine,
        current_step_id: StateId | None = None,
    ) -> None:
        super().__init__(state_machine, current_step_id)
        self.prop_name = ""
        self.prop_value: Any | None = None

    def clone(self) -> KeyValueStepper:
        cloned_stepper = super().clone()
        cloned_stepper.prop_name = self.prop_name
        cloned_stepper.prop_value = self.prop_value
        return cloned_stepper

    def should_complete_step(self) -> bool:
        """
        Handle the completion of a transition by setting the property name and value.

        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if not super().should_complete_step() or not self.sub_stepper:
            return False

        try:
            if self.target_state == 1:
                self.prop_name = json.loads(self.sub_stepper.get_raw_value())
            elif self.target_state in self.state_machine.end_states:
                self.prop_value = json.loads(self.sub_stepper.get_raw_value())
        except Exception:
            return False

        return True

    def get_current_value(self) -> tuple[str, Any]:
        """
        Get the parsed property as a key-value pair.

        Returns:
            Tuple[str, Any]: A tuple containing the property name and its corresponding value.
        """
        if self.prop_name is None:
            return ("", None)
        return (self.prop_name, self.prop_value)
