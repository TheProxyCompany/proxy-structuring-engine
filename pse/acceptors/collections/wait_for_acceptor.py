from __future__ import annotations
from typing import Iterable, Optional, Callable, Type
import logging

from pse.acceptors.basic.acceptor import Acceptor
from pse.core.state_machine import StateMachine
from pse.core.walker import Walker

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
        self.start_hook = start_hook
        self.end_hook = end_hook
        self.triggered = False

    @property
    def walker_class(self) -> Type[Walker]:
        return WaitForWalker

    def advance(self, walker: Walker, token: str) -> Iterable[Walker]:
        return self.wait_for_acceptor.advance(walker, token)

    def __repr__(self) -> str:
        return f"WaitForAcceptor({repr(self.wait_for_acceptor)})"


class WaitForWalker(Walker):
    """
    Walker for handling the WaitForAcceptor.
    Manages internal walkers that monitor for the triggering acceptor.
    """

    def __init__(
        self,
        acceptor: WaitForAcceptor,
        walkers: Optional[Iterable[Walker]] = None,
    ):
        """
        Initialize the WaitForAcceptor Walker.

        Args:
            acceptor (WaitForAcceptor): The parent WaitForAcceptor.
            walkers (Optional[Iterable[StateMachineAcceptor.Walker]], optional):
                Existing walkers to manage. Defaults to those from the wait_for_acceptor.
        """
        super().__init__(acceptor)
        self.acceptor = acceptor
        if walkers:
            self.walkers = list(walkers)
        else:
            self.walkers = list(self.acceptor.wait_for_acceptor.get_walkers())

    def should_start_transition(self, token: str) -> bool:
        for walker in self.walkers:
            if walker.should_start_transition(token):
                return True

        return False

    def should_complete_transition(self) -> bool:
        for walker in self.walkers:
            if walker.should_complete_transition():
                return True

        return False

    def accepts_any_token(self) -> bool:
        """
        Indicates that this acceptor matches all characters until a trigger is found.

        Returns:
            bool: Always True.
        """
        return True

    def can_accept_more_input(self) -> bool:
        return False

    def is_within_value(self) -> bool:
        """
        Determine if the walker is currently within a value.

        Returns:
            bool: True if in a value, False otherwise.
        """
        return self.acceptor.triggered or any(
            walker.is_within_value() for walker in self.walkers
        )

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advance all internal walkers with the given input.

        Args:
            input (str): The input to process.

        Returns:
            Iterable[TokenAcceptor.Walker]: Updated walkers after processing.
        """
        new_walkers = []

        for advanced_token, walker in self.acceptor.advance_all(self.walkers, token):
            if walker.has_reached_accept_state():
                self.acceptor.triggered = True
                if self.acceptor.end_hook:
                    self.acceptor.end_hook()
                yield walker
                return
            else:
                new_walkers.append(walker)

        yield WaitForWalker(self.acceptor, new_walkers)

    @property
    def current_value(self) -> str:
        """
        Retrieve the current value indicating the wait state.

        Returns:
            str: Description of the waiting state.
        """
        return "\n".join(repr(walker) for walker in self.walkers)
