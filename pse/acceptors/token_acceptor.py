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
from abc import ABC, abstractmethod
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

    @property
    def is_optional(self) -> bool:
        """Checks if the acceptor is optional.

        Returns:
            bool: True if the acceptor is optional, False otherwise.
        """
        return self._is_optional

    @property
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
    def get_transition_walkers(
        self,
        walker: Walker,
    ) -> Iterable[Tuple[Walker, StateType]]:
        """Retrieves transitions from the given walker."""
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

    @abstractmethod
    def branch_walker(
        self,
        walker: Walker,
        token: Optional[str] = None,
    ) -> Iterable[Walker]:
        """Branch the walker into multiple paths for parallel exploration."""
        pass

    #
    # String representations
    #

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Return a formatted string representation of the StateMachine instance.

        This method provides a detailed view of the state machine's configuration,
        formatted with proper indentation for better readability.

        Returns:
            str: A formatted string showing the state machine's configuration.
        """

        def format_graph(graph: StateMachineGraph, indent: int = 0) -> str:
            if not graph:
                return ""

            lines = []
            indent_str = "    " * indent
            lines.append("graph={\n")
            for state, transitions in graph.items():
                lines.append(f"{indent_str}    {state}: [")
                transition_lines = []
                for acceptor, target_state in transitions:
                    acceptor_repr = format_acceptor(acceptor, indent + 2)
                    target_state_str = (
                        "'$'" if target_state == "$" else str(target_state)
                    )
                    transition_lines.append(f"({acceptor_repr}, {target_state_str})")
                lines.append(", ".join(transition_lines) + "],\n")
            lines.append(f"{indent_str}}}")
            return "".join(lines)

        def format_acceptor(acceptor: TokenAcceptor, indent: int) -> str:
            acceptor_repr = acceptor.__repr__()
            return "\n".join(
                ("    " * indent + line) if idx != 0 else line
                for idx, line in enumerate(acceptor_repr.splitlines())
            )

        formatted_graph = format_graph(self.graph)
        return f"{self.__class__.__name__}({formatted_graph})"
