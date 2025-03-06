from __future__ import annotations

import logging
from typing import Self

from pse_core import StateGraph
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.types.whitespace import WhitespaceStateMachine

logger = logging.getLogger(__name__)

class LoopStateMachine(StateMachine):
    """
    Loop through a single StateMachine.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        min_loop_count: int = 1,
        max_loop_count: int = -1,
        whitespace_seperator: bool = False,
    ) -> None:
        """
        Args:
            state_machine: State machine to be looped through
        """
        self.whitespace_seperator = WhitespaceStateMachine() if whitespace_seperator else None
        if self.whitespace_seperator:
            state_graph: StateGraph = {
                0: [(state_machine, 1)],
                1: [(self.whitespace_seperator, 2)],
                2: [(state_machine, 1)],
            }
        else:
            state_graph: StateGraph = {
                0: [(state_machine, 1)],
                1: [(state_machine, 0)],
            }

        super().__init__(
            state_graph=state_graph,
            is_optional=min_loop_count == 0,
        )
        self.min_loop_count = min_loop_count or 1
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
        """
        Determines if this stepper is in an accept state.
        
        A loop stepper is in an accept state if:
        1. It has completed at least the minimum required iterations, and
        2. If it's currently processing a sub-stepper, that sub-stepper is in an accept state
        
        Returns:
            True if the stepper is in an accept state, False otherwise
        """
        # Early exit if we haven't reached the minimum loop count
        if self.loop_count < self.state_machine.min_loop_count:
            return False
            
        # If we're currently processing a value through a sub-stepper,
        # delegate to that sub-stepper's accept state
        if self.sub_stepper is not None and self.sub_stepper.is_within_value():
            return self.sub_stepper.has_reached_accept_state()
            
        # Otherwise, we're in an accept state if we've met the minimum loop count
        return True

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
        if (
            self.state_machine.whitespace_seperator
            and stepper.state_machine == self.state_machine.whitespace_seperator
        ):
            return

        self.loop_count += 1
        return super().add_to_history(stepper)

    def get_final_state(self) -> list[Stepper]:
        """
        Gets the final state representation for this stepper.
        
        This method decides which stepper(s) to return as the final state:
        - If we're within a value in the sub-stepper (and it's not a whitespace separator),
          delegate to the sub-stepper's final state
        - Otherwise, return this stepper's history
        
        Returns:
            List of steppers representing the final state
        """
        # Cache whitespace separator for performance
        whitespace_sep = self.state_machine.whitespace_seperator
        
        # Delegate to the sub-stepper if:
        # 1. We have a sub-stepper that's currently processing a value
        # 2. The sub-stepper is not a whitespace separator
        if (self.sub_stepper and 
            self.sub_stepper.is_within_value() and 
            not (whitespace_sep and self.sub_stepper.state_machine == whitespace_sep)):
            return self.sub_stepper.get_final_state()

        # Otherwise, return this stepper's history
        return self.history
