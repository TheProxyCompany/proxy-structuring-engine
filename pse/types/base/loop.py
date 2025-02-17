from __future__ import annotations

import logging
from typing import Self

from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

logger = logging.getLogger(__name__)

class LoopStateMachine(StateMachine):
    """
    Loop through a single StateMachine.
    """

    def __init__(self, state_machine: StateMachine, min_loop_count: int = 1, max_loop_count: int = -1) -> None:
        """
        Args:
            state_machine: State machine to be looped through
        """
        super().__init__(
            state_graph={
                0: [(state_machine, 1)],
                1: [(state_machine, 0)],
            }
        )
        self.min_loop_count = min_loop_count
        self.max_loop_count = max_loop_count

    def get_new_stepper(self, state: int | str | None = None) -> Stepper:
        return LoopStepper(self, state)

    def __str__(self) -> str:
        return "Loop"

class LoopStepper(Stepper):
    """
    A stepper that loops through a single StateMachine.
    """

    def __init__(self, loop_state_machine: LoopStateMachine, *args, **kwargs) -> None:
        super().__init__(loop_state_machine, *args, **kwargs)
        self.state_machine: LoopStateMachine = loop_state_machine
        self.loop_count = 0

    def clone(self) -> Self:
        clone = super().clone()
        clone.loop_count = self.loop_count
        return clone

    def has_reached_accept_state(self) -> bool:
        if self.loop_count >= self.state_machine.min_loop_count:
            if self.sub_stepper is not None and self.sub_stepper.is_within_value():
                return self.sub_stepper.has_reached_accept_state()
            return True
    
        return False

    def can_accept_more_input(self) -> bool:
        if not super().can_accept_more_input():
            return False

        if self.state_machine.max_loop_count > 0:
            return self.loop_count < self.state_machine.max_loop_count

        return True

    def should_start_step(self, token: str) -> bool:
        if self.loop_count >= self.state_machine.max_loop_count:
            return False

        return super().should_start_step(token)

    def add_to_history(self, stepper: Stepper) -> None:
        self.loop_count += 1
        return super().add_to_history(stepper)
