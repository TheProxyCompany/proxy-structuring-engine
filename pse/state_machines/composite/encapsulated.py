from __future__ import annotations

from typing import Any, Self

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.wait_for import WaitFor


class EncapsulatedStateMachine(StateMachine):
    """
    Accepts JSON data within a larger text, delimited by specific markers.

    This class encapsulates an state_machine that recognizes JSON content framed by
    specified opening and closing delimiters. Stores text before the delimiters in the WaitForStateMachine's buffer.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        delimiters: tuple[str, str],
        min_buffer_length: int = -1,
    ) -> None:
        """

        Args:
            state_machine: The state_machine responsible for validating the JSON content.
            open_delimiter: The string that denotes the start of the JSON content.
            close_delimiter: The string that denotes the end of the JSON content.
        """
        super().__init__(
            {
                0: [
                    (
                        WaitFor(
                            PhraseStateMachine(delimiters[0]),
                            min_buffer_length=min_buffer_length,
                        ),
                        1,
                    ),
                ],
                1: [(state_machine, 2)],
                2: [(PhraseStateMachine(delimiters[1]), "$")],
            }
        )
        self.inner_state_machine = state_machine
        self.delimiters = delimiters

    def get_new_stepper(self, state: StateId | None = None) -> EncapsulatedStepper:
        return EncapsulatedStepper(self, state)

    def __str__(self) -> str:
        components = []
        if self.delimiters[0]:
            components.append(f"{self.delimiters[0]!r}")

        if self.inner_state_machine:
            components.append(str(self.inner_state_machine))

        if self.delimiters[1]:
            components.append(f"{self.delimiters[1]!r}")
        return f"Encapsulated({', '.join(components)})"


class EncapsulatedStepper(Stepper):

    def __init__(self, state_machine: EncapsulatedStateMachine, state: StateId | None = None) -> None:
        super().__init__(state_machine, state)
        self.state_machine: EncapsulatedStateMachine = state_machine
        self.scratch_pad = ""
        self.inner_stepper: Stepper | None = None

    def clone(self) -> Self:
        clone = super().clone()
        clone.scratch_pad = self.scratch_pad
        clone.inner_stepper = self.inner_stepper.clone() if self.inner_stepper else None
        return clone

    def accepts_any_token(self) -> bool:
        if self.sub_stepper:
            within_value = self.sub_stepper.accepts_any_token()
            return within_value

        return self.current_state == 0

    def get_invalid_continuations(self) -> list[str]:
        if self.current_state == 0 and self.sub_stepper:
            return self.sub_stepper.get_invalid_continuations()
        return []

    def is_within_value(self) -> bool:
        if self.current_state == 0 and self.sub_stepper:
            return self.sub_stepper.is_within_value()

        return self.current_state != 0

    def add_to_history(self, stepper: Stepper) -> None:
        if self.current_state == 2:
            self.inner_stepper = stepper

        return super().add_to_history(stepper)

    def get_current_value(self) -> tuple[str, Any]:
        if self.inner_stepper:
            return self.inner_stepper.get_current_value()

        return super().get_current_value()
