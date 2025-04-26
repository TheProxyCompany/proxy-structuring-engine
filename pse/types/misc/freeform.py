from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pse_core import StateId

from pse.types.base.wait_for import WaitFor, WaitForStepper
from pse.types.enum import EnumStateMachine


class FreeformStateMachine(WaitFor):
    """
    A state machine that can be used to parse freeform text that has an ending delimiter.
    """

    def __init__(
        self,
        end_delimiters: list[str],
        char_min: int | None = None,
    ):
        self.end_delimiters = end_delimiters
        delimiter_state_machine = EnumStateMachine(self.end_delimiters, require_quotes=False)
        super().__init__(
            delimiter_state_machine,
            buffer_length=char_min or 1,
        )

    def get_new_stepper(self, _: StateId | None = None) -> FreeformStepper:
        return FreeformStepper(self)

    def __str__(self) -> str:
        return "FreeformText"

class FreeformStepper(WaitForStepper):

    def __init__(
        self,
        state_machine: FreeformStateMachine,
    ):
        super().__init__(state_machine)
        self.state_machine: FreeformStateMachine = state_machine

    def get_raw_value(self) -> str:
        """
        Get the raw value of the buffer.
        """
        if self.sub_stepper:
            return self.buffer + self.sub_stepper.get_raw_value()
        elif self.history:
            accepted_raw_value = self.history[-1].get_raw_value()
            return self.buffer + accepted_raw_value

        return self.buffer

    def get_current_value(self) -> Any:
        return self.buffer

    def get_token_safe_output(
        self,
        decode_function: Callable[[list[int]], str],
    ) -> str:
        """
        Get the token safe output of the buffer.
        """
        safe_output = super().get_token_safe_output(decode_function)
        for end_delimiter in self.state_machine.end_delimiters:
            if safe_output.endswith(end_delimiter):
                return safe_output[:-len(end_delimiter)]

        return safe_output
