from __future__ import annotations

from collections.abc import Callable
from typing import Self

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

    def get_final_state(self) -> list[Stepper]:
        return [self]

    def get_token_safe_output(self, decode_function: Callable[[list[int]], str]) -> str:
        """
        Retrieve the token-safe output with delimiters removed.

        This method processes the raw output by removing the encapsulating delimiters,
        handling both complete and partial delimiter occurrences efficiently.

        Args:
            decode_function: Function to decode token IDs into a string

        Returns:
            Processed string with delimiters stripped
        """
        # Get and decode the token history
        token_ids = self.get_token_ids_history()
        token_safe_output: str = decode_function(token_ids).strip()
        
        # Extract delimiters
        start_delim, end_delim = self.state_machine.delimiters

        # Remove start delimiter - optimize by checking exact match first
        # This is faster than always using lstrip
        if token_safe_output.startswith(start_delim):
            token_safe_output = token_safe_output[len(start_delim):]
        else:
            token_safe_output = token_safe_output.lstrip(start_delim)

        # Remove end delimiter - optimize by checking exact match first
        # This is faster than always using rstrip
        if token_safe_output.endswith(end_delim):
            token_safe_output = token_safe_output[:-len(end_delim)]
        else:
            token_safe_output = token_safe_output.rstrip(end_delim)

        return token_safe_output
