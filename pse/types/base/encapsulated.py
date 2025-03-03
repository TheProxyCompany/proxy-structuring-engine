from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.types.base.phrase import PhraseStateMachine
from pse.types.base.wait_for import WaitFor


class EncapsulatedStateMachine(StateMachine):
    """
    This class encapsulates an state_machine that recognizes content framed by
    specified opening and closing delimiters.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        delimiters: tuple[str, str] | None,
        buffer_length: int = -1,
        is_optional: bool = False,
    ) -> None:
        """

        Args:
            state_machine: The state_machine wrapped by this state machine.
            delimiters: The tuple of opening and closing delimiters.
        """
        self.inner_state_machine = state_machine
        self.delimiters = delimiters or ("```", "```")
        super().__init__(
            {
                0: [
                    (
                        WaitFor(
                            PhraseStateMachine(self.delimiters[0]),
                            buffer_length=buffer_length,
                        ),
                        1,
                    ),
                ],
                1: [(state_machine, 2)],
                2: [(PhraseStateMachine(self.delimiters[1]), "$")],
            },
            is_optional=is_optional,
        )

    def get_new_stepper(self, state: StateId | None = None) -> EncapsulatedStepper:
        return EncapsulatedStepper(self, state)

class EncapsulatedStepper(Stepper):

    def __init__(
        self,
        state_machine: EncapsulatedStateMachine,
        state: StateId | None = None,
    ) -> None:
        super().__init__(state_machine, state)
        self.state_machine: EncapsulatedStateMachine = state_machine
        self.inner_stepper: Stepper | None = None

    def clone(self) -> Self:
        clone = super().clone()
        if self.inner_stepper:
            clone.inner_stepper = self.inner_stepper.clone()
        return clone

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

    def get_token_safe_output(self, decode_function: Callable[[list[int]], str]) -> Any:
        """
        Get the token safe output from the inner stepper.
        This function strips the delimiters from the output, gracefully handling partial or malformed delimiters.
        """
        token_safe_output: str = super().get_token_safe_output(decode_function)

        start_delim, end_delim = self.state_machine.delimiters

        # Remove starting delimiter fragments from head, preserving content whitespace
        start_index = 0
        for i in range(len(start_delim), 0, -1):
            if token_safe_output.startswith(start_delim[:i]):
                start_index = i
                break

        # Find the end of the starting delimiter, handling partial matches
        if start_index > 0:
            token_safe_output = token_safe_output[start_index:]

        # Remove ending delimiter fragments from tail, preserving content whitespace
        end_index = len(token_safe_output)
        for i in range(len(end_delim), 0, -1):
            if token_safe_output.endswith(end_delim[-i:]):
                end_index = len(token_safe_output) - i
                break

        # Find the start of the ending delimiter, handling partial matches
        if end_index < len(token_safe_output):
            token_safe_output = token_safe_output[:end_index]

        return token_safe_output
