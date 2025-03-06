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
        buffer_length: int = -1,
        strict: bool = True,
    ):
        """
        Initialize with a target nested StateId Machine.

        Args:
            state_machine (StateMachine): The nested StateId Machine to watch for.
            min_buffer_length (int):
                The minimum length of the buffer
            strict (bool):
                If True, the nested StateId Machine's progress is reset
                when invalid input is detected.
        """
        super().__init__()

        self.min_buffer_length = buffer_length
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
        Determines if this stepper can accept any token based on buffer state.
        
        The stepper accepts any token if:
        1. The buffer meets minimum length requirements, or
        2. The sub-stepper is active and accepts any token
        
        Returns:
            True if the stepper can accept any token, False otherwise
        """
        # Cache min_buffer_length for performance
        min_buffer_length = self.state_machine.min_buffer_length
        
        # Delegate to sub_stepper if it's active
        if self.sub_stepper and (
            self.sub_stepper.is_within_value() or min_buffer_length == -1
        ):
            return self.sub_stepper.accepts_any_token()

        # Otherwise, check buffer length
        return len(self.buffer) >= min_buffer_length

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
        """
        Determines if the stepper should start processing the token.
        
        This method decides whether to start a step based on:
        1. Whether we have remaining input from a previous token
        2. The buffer length requirements
        3. The stepper's current state
        
        Args:
            token: The token to potentially process
            
        Returns:
            True if the step should start, False otherwise
        """
        # Never start a step if we have remaining input
        if self.remaining_input:
            return False

        # Cache frequently accessed values
        required_buffer_length = self.state_machine.min_buffer_length
        should_start = super().should_start_step(token)
        
        # Handle unlimited buffer length case
        if required_buffer_length <= 0:
            return should_start or not self.is_within_value()
            
        # For cases with a positive buffer length requirement
        buffer_length = len(self.buffer)
        is_in_value = self.is_within_value()
        
        # Return True if either:
        # 1. super().should_start_step() returns True and we have enough buffer, or
        # 2. super().should_start_step() returns False but we're not within a value
        if should_start and buffer_length >= required_buffer_length:
            return True
        if not should_start and not is_in_value:
            return True
            
        return False

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

        if invalid_prefix and (
            not self.is_within_value() or not self.state_machine.strict
        ):
            if not self.is_within_value() and self.state_machine.min_buffer_length == -1:
                return []

            clone = self.clone()
            clone.buffer += invalid_prefix
            if valid_suffix:
                return self.state_machine.advance_stepper(clone, valid_suffix)
            else:
                return [clone]

        return self.state_machine.advance_stepper(self, valid_suffix)
