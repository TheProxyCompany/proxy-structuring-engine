from __future__ import annotations

import logging
from typing import Any, Self

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

logger = logging.getLogger(__name__)


class WaitForStateMachine(StateMachine):
    """
    Accept all text until a segment triggers another specified state_machine.

    This is particularly useful for allowing free-form text until a specific
    delimiter or pattern is detected, such as when parsing output from
    language models that encapsulate JSON within markdown code blocks.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        allow_break: bool = False,
        min_length_before_trigger: int = 0,
    ):
        """
        Initialize the WaitForAcceptor with a target state_machine to watch for.

        Args:
            wait_for_acceptor (TokenAcceptor): The state_machine that, when triggered,
                stops the waiting and stops accepting further characters.
        """
        super().__init__()
        self.wait_for_sm = state_machine
        self.allow_break = allow_break
        self.min_length_before_trigger = min_length_before_trigger

    def get_transitions(self, walker: Walker) -> list[tuple[Walker, State]]:
        """
        Get transitions for the WaitForAcceptor.
        """
        transitions = []
        for transition in self.wait_for_sm.get_walkers():
            transitions.append((transition, "$"))
        return transitions

    def get_walkers(self, state: State | None = None) -> list[Walker]:
        """
        return:
            Walkers for the WaitForAcceptor.
        """
        return self.branch_walker(WaitForWalker(self))

    def __str__(self) -> str:
        return f"WaitFor({self.wait_for_sm})"


class WaitForWalker(Walker):
    """
    Walker for handling the WaitForAcceptor.
    Manages internal walkers that monitor for the triggering state_machine.
    """

    def __init__(self, state_machine: WaitForStateMachine):
        """
        Initialize the WaitForAcceptor Walker.

        Args:
            state_machine (WaitForAcceptor): The parent WaitForAcceptor.
        """
        super().__init__(state_machine)
        self.target_state = "$"
        self.state_machine: WaitForStateMachine = state_machine
        self.before_trigger = ""

    def clone(self) -> Self:
        clone = super().clone()
        clone.before_trigger = self.before_trigger
        return super().clone()

    def should_start_transition(self, token: str) -> bool:
        if self.state_machine.min_length_before_trigger > 0:
            if len(self.get_raw_value()) < self.state_machine.min_length_before_trigger:
                return False

        if self.transition_walker and self.transition_walker.is_within_value():
            return self.transition_walker.should_start_transition(token)

        return True

    def accepts_any_token(self) -> bool:
        """
        Indicates that this state_machine matches all characters until a trigger is found.

        Returns:
            bool: Always True.
        """
        if self.transition_walker and self.transition_walker.is_within_value():
            return self.transition_walker.accepts_any_token()
        return True

    def consume(self, token: str) -> list[Walker]:
        """
        Advance all internal walkers with the given input.

        Args:
            input (str): The input to process.

        Returns:
            list[TokenAcceptor.Walker]: Updated walkers after processing.
        """
        if self.transition_walker:
            if not self.transition_walker.should_start_transition(token):
                new_walkers = []
                self.before_trigger += token
                if self.state_machine.allow_break or not self.is_within_value():
                    new_walkers.append(self)
                else:
                    for transition in self.state_machine.wait_for_sm.get_walkers():
                        if new_walker := self.start_transition(transition):
                            new_walker.target_state = "$"
                            new_walkers.append(new_walker)
                return new_walkers

        return self.state_machine.advance_walker(self, token)

    def get_current_value(self) -> tuple[str, Any]:
        if self.transition_walker:
            return self.before_trigger, self.transition_walker.get_current_value()

        return self.before_trigger, None
