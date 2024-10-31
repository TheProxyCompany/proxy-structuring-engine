from __future__ import annotations

from pse.state_machine.walker import Walker
from pse.state_machine.state_machine import StateMachine
from typing import Iterable

class EmptyTransitionAcceptor(StateMachine):
    """
    Faux acceptor that allows the creation of empty transition edges in a state machine graph.

    This facilitates the expression of complex graphs by skipping the current state without consuming input.
    """

    def get_walkers(self) -> Iterable[Walker]:
        return []

    def is_optional(self) -> bool:
        return True

EmptyTransition = EmptyTransitionAcceptor({})

__all__ = [
    "EmptyTransition",
]
