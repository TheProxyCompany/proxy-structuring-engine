"""
Base Token Acceptors Module.

This module defines the foundational classes and methods for token acceptors,
which constrain the tokens acceptable during parsing or generation of text.
Token acceptors utilize walkers to manage multiple parsing states efficiently,
minimizing expensive backtracking operations.

Classes:
    TokenAcceptor: Base class for all token acceptors.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Type

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pse.state_machine.walker import Walker
    from pse.state_machine.types import StateType, StateMachineGraph


class TokenAcceptor(ABC):
    """
    Base class for token acceptors.

    A token acceptor constrains the acceptable tokens at a specific point
    during parsing or generation.

    It manages multiple walkers representing different valid states,
    enabling efficient traversal and minimizing backtracking.

    Attributes:
        initial_state (StateType): The starting state of the acceptor.
        end_states (Iterable[StateType]): A collection of acceptable end states.
    """

    _MAX_WORKERS: int = min(32, os.cpu_count() or 1)
    _EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

    def __init__(
        self,
        graph: StateMachineGraph,
        initial_state: StateType,
        end_states: Iterable[StateType],
        walker_type: Type[Walker],
        is_optional: bool = False,
        is_case_sensitive: bool = True,
    ) -> None:
        """Initializes the TokenAcceptor with the given initial and end states.

        Args:
            initial_state (StateType): The starting state of the acceptor.
            end_states (Iterable[StateType]): A collection of acceptable end states.
        """
        self.graph = graph
        self.initial_state = initial_state
        self.end_states = end_states
        self._walker = walker_type
        self._is_optional = is_optional
        self._is_case_sensitive = is_case_sensitive

    def is_optional(self) -> bool:
        """Checks if the acceptor is optional.

        Returns:
            bool: True if the acceptor is optional, False otherwise.
        """
        return self._is_optional

    def is_case_sensitive(self) -> bool:
        """Checks if the acceptor is case sensitive.

        Returns:
            bool: True if the acceptor is case sensitive, False otherwise.
        """
        return self._is_case_sensitive

    @abstractmethod
    def get_walkers(
        self,
        state: Optional[StateType] = None,
    ) -> Iterable[Walker]:
        """Retrieves walkers to traverse the acceptor.

        Returns:
            Iterable[Walker]: An iterable of walker instances.
        """
        pass

    @abstractmethod
    def get_transitions(
        self,
        walker: Walker,
    ) -> Iterable[Tuple[Walker, StateType]]:
        """Retrieves transitions from the given walker."""
        pass

    @abstractmethod
    def branch_walker(
        self,
        walker: Walker,
        token: Optional[str] = None,
    ) -> Iterable[Walker]:
        """Branch the walker into multiple paths for parallel exploration."""
        pass

    @abstractmethod
    def advance_walker(
        self,
        walker: Walker,
        token: str,
    ) -> Iterable[Walker]:
        """Advances the walker with the given input.

        Args:
            walker (Walker): The walker to advance.
            token (str): The input string to process.

        Returns:
            Iterable[Walker]: An iterable of updated walkers after advancement.
        """
        pass
