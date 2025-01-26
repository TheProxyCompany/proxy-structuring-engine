from __future__ import annotations

from typing import Any

from pse.state_machines.base.character import (
    CharacterStateMachine,
    CharacterStepper,
)


class IntegerStateMachine(CharacterStateMachine):
    """
    Accepts an integer as per JSON specification.
    """

    def __init__(self, drop_leading_zeros: bool = True) -> None:
        super().__init__("0123456789")
        self.drop_leading_zeros = drop_leading_zeros

    def get_new_stepper(self, state: int | str) -> IntegerStepper:
        return IntegerStepper(self)

    def __str__(self) -> str:
        return "Integer"


class IntegerStepper(CharacterStepper):

    def __init__(
        self, state_machine: IntegerStateMachine, value: str | None = None
    ) -> None:
        super().__init__(state_machine, value)
        self.state_machine: IntegerStateMachine = state_machine

    def get_current_value(self) -> Any:
        if self.state_machine.drop_leading_zeros and self._raw_value is not None:
            return int(self._raw_value)
        return self._raw_value
