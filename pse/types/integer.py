from __future__ import annotations

from typing import Any

from pse.types.base.character import (
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
        if self._raw_value is None:
            return None
        return int(self._raw_value) if self.state_machine.drop_leading_zeros else self._raw_value
