from __future__ import annotations
from typing import Iterable, Optional, Callable, Tuple

from pse.acceptors.basic.acceptor import Acceptor
from pse.core.state_machine import StateMachine, State
from pse.core.walker import Walker
import logging
logger = logging.getLogger(__name__)


class WaitForAcceptor(StateMachine):
    """
    Accept all text until a segment triggers another specified acceptor.

    This is particularly useful for allowing free-form text until a specific
    delimiter or pattern is detected, such as when parsing output from
    language models that encapsulate JSON within markdown code blocks.
    """

    def __init__(
        self,
        wait_for_acceptor: Acceptor,
        allow_break: bool = False,
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
    ):
        """
        Initialize the WaitForAcceptor with a target acceptor to watch for.

        Args:
            wait_for_acceptor (TokenAcceptor): The acceptor that, when triggered,
                stops the waiting and stops accepting further characters.
        """
        super().__init__()
        self.wait_for_acceptor = wait_for_acceptor
        self.allow_break = allow_break
        self.start_hook = start_hook
        self.end_hook = end_hook
        self.triggered = False

    def get_transitions(
        self, walker: Walker, state: Optional[State] = None
    ) -> Iterable[Tuple[Walker, State, State]]:
        """
        Get transitions for the WaitForAcceptor.
        """
        for transition in self.wait_for_acceptor.get_walkers():
            yield transition, 0, "$"

    def get_walkers(self) -> Iterable[Walker]:
        """
        Yields:
            Walkers for the WaitForAcceptor.
        """
        yield from self.branch_walker(WaitForWalker(self))


class WaitForWalker(Walker):
    """
    Walker for handling the WaitForAcceptor.
    Manages internal walkers that monitor for the triggering acceptor.
    """

    def __init__(self, acceptor: WaitForAcceptor):
        """
        Initialize the WaitForAcceptor Walker.

        Args:
            acceptor (WaitForAcceptor): The parent WaitForAcceptor.
        """
        super().__init__(acceptor)
        self.target_state = "$"
        self.acceptor = acceptor

    def should_start_transition(self, token: str) -> bool:
        if self.transition_walker and self.transition_walker.is_within_value():
            return self.transition_walker.should_start_transition(token)
        return True

    def accepts_any_token(self) -> bool:
        """
        Indicates that this acceptor matches all characters until a trigger is found.

        Returns:
            bool: Always True.
        """
        if self.transition_walker and self.transition_walker.is_within_value():
            return False
        return True

    def can_accept_more_input(self) -> bool:
        return False

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

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advance all internal walkers with the given input.

        Args:
            input (str): The input to process.

        Returns:
            Iterable[TokenAcceptor.Walker]: Updated walkers after processing.
        """
        if (
            not self.transition_walker
            or not self.transition_walker.should_start_transition(token)
        ):
            # wait for acceptor blindly accepts all tokens while
            # trying to advance the trigger acceptor
            if not self.acceptor.allow_break:
                self.transition_walker = None
                yield from self.branch()
                return

            yield self
            return

        yield from self.acceptor.advance(self, token)
