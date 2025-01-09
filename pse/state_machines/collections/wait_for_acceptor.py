from __future__ import annotations

import logging
from collections.abc import Callable

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

logger = logging.getLogger(__name__)


class WaitForAcceptor(StateMachine):
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
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
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
        self.start_hook = start_hook
        self.end_hook = end_hook
        self.triggered = False

    def get_transitions(
        self, walker: Walker, state: State | None = None
    ) -> list[tuple[Walker, State, State]]:
        """
        Get transitions for the WaitForAcceptor.
        """
        transitions = []
        for transition in self.wait_for_sm.get_walkers():
            transitions.append((transition, 0, "$"))
        return transitions

    def get_walkers(self, state: State | None = None) -> list[Walker]:
        """
        return:
            Walkers for the WaitForAcceptor.
        """
        return self.branch_walker(WaitForWalker(self))


class WaitForWalker(Walker):
    """
    Walker for handling the WaitForAcceptor.
    Manages internal walkers that monitor for the triggering state_machine.
    """

    def __init__(self, state_machine: WaitForAcceptor):
        """
        Initialize the WaitForAcceptor Walker.

        Args:
            state_machine (WaitForAcceptor): The parent WaitForAcceptor.
        """
        super().__init__(state_machine)
        self.target_state = "$"
        self.state_machine: WaitForAcceptor = state_machine

    def should_start_transition(self, token: str) -> bool:
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

    def can_accept_more_input(self) -> bool:
        return False

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        if self.transition_walker and self.transition_walker.is_within_value():
            return self.transition_walker.get_valid_continuations(depth)
        return [] # any token is valid

    def is_within_value(self) -> bool:
        """
        Determine if the walker is currently within a value.

        Returns:
            bool: True if in a value, False otherwise.
        """
        return (
            self.transition_walker is not None
            and self.transition_walker.is_within_value()
        )

    def consume_token(self, token: str) -> list[Walker]:
        """
        Advance all internal walkers with the given input.

        Args:
            input (str): The input to process.

        Returns:
            list[TokenAcceptor.Walker]: Updated walkers after processing.
        """
        if self.transition_walker:
            if not self.transition_walker.should_start_transition(token):
                if self.state_machine.allow_break:
                    return [self]
                else:
                    self.transition_walker = None
                    return self.branch()

        return self.state_machine.advance(self, token)
