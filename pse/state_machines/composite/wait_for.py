from __future__ import annotations

import logging
from typing import Self

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

logger = logging.getLogger(__name__)


class WaitFor(StateMachine):
    """
    Accept all text until a segment triggers a nested StateId Machine.

    Accumulates text in a buffer until a segment triggers the nested StateId Machine.

    This is particularly useful for allowing free-form text until a specific
    delimiter or pattern is detected, such as when parsing output from
    language models that encapsulate JSON within markdown code blocks.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        min_buffer_length: int = -1,
        strict: bool = True,
    ):
        """
        Initialize with a target nested StateId Machine.

        Args:
            state_machine (StateMachine): The nested StateId Machine to watch for.
            min_buffer_length (int): The minimum length of the buffer before the nested StateId Machine can accept input.
            strict (bool): If True, the nested StateId Machine's progress is reset when invalid input is detected.
        """
        super().__init__()

        self.min_buffer_length = min_buffer_length
        self.strict = strict
        self.wait_for_sm = state_machine

    def get_transitions(self, stepper: Stepper) -> list[tuple[Stepper, StateId]]:
        transitions = []
        for transition in self.wait_for_sm.get_steppers():
            transitions.append((transition, "$"))
        return transitions

    def get_steppers(self, state: StateId | None = None) -> list[Stepper]:
        return self.branch_stepper(WaitForStepper(self))

    def __str__(self) -> str:
        return f"WaitFor({self.wait_for_sm})"


class WaitForStepper(Stepper):
    def __init__(self, state_machine: WaitFor):
        super().__init__(state_machine)
        self.target_state = "$"
        self.state_machine: WaitFor = state_machine
        self.buffer = ""

    def clone(self) -> Self:
        clone = super().clone()
        clone.buffer = self.buffer
        return clone

    def accepts_any_token(self) -> bool:
        """
        Indicates that this state_machine matches all characters until a trigger is found.

        Returns:
            bool: Always True.
        """
        if self.sub_stepper and (self.sub_stepper.is_within_value() or self.state_machine.min_buffer_length == -1):
            return self.sub_stepper.accepts_any_token()

        return len(self.buffer) >= self.state_machine.min_buffer_length

    def get_valid_continuations(self) -> list[str]:
        """
        If the buffer is long enough, we can accept any valid continuations.

        If the buffer is not long enough, we can accept everything.
        """
        if len(self.buffer) >= self.state_machine.min_buffer_length:
            return super().get_valid_continuations()
        return []

    def get_invalid_continuations(self) -> list[str]:
        """
        If the buffer is not long enough yet,
        any valid continuation is inversed and
        invalid to allow the buffer to grow.

        If the buffer is long enough, there are no invalid continuations.
        """
        if len(self.buffer) < self.state_machine.min_buffer_length and self.sub_stepper:
            return self.sub_stepper.get_valid_continuations()
        return []

    def should_start_step(self, token: str) -> bool:
        if self.has_reached_accept_state():
            return self.remaining_input is None

        required_buffer_length = self.state_machine.min_buffer_length
        should_start = super().should_start_step(token)
        if required_buffer_length > 0:
            if should_start and len(self.buffer) >= required_buffer_length:
                # we have enough characters to start the transition
                return True
            elif not should_start and not self.is_within_value():
                # in this case, we are not within a value, so we can start the transition
                # to allow the buffer/scratchpad to grow
                return True
            else:
                # we don't have enough characters to start the transition
                return False

        # if we don't have a required buffer length, we can start the transition
        return should_start or not self.is_within_value()

    def consume(self, token: str) -> list[Stepper]:
        # No sub_stepper means we can't process anything
        if not self.sub_stepper:
            return []

        # Try to find the longest valid prefix that the sub_stepper will accept
        invalid_prefix = ""
        valid_suffix = token

        while valid_suffix and not self.sub_stepper.should_start_step(valid_suffix):
            invalid_prefix += valid_suffix[0]
            valid_suffix = valid_suffix[1:]

        if self.state_machine.strict and self.is_within_value() and invalid_prefix:
            return []

        if invalid_prefix and (not self.is_within_value() or not self.state_machine.strict):
            clone = self.clone()
            clone.buffer += invalid_prefix
            if valid_suffix:
                return self.state_machine.advance_stepper(clone, valid_suffix)
            else:
                return [clone]
        # Process the valid prefix normally
        return self.state_machine.advance_stepper(self, valid_suffix)
