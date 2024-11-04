"""State machine walker module.

This module provides the base `Walker` class for traversing state machines during parsing
and generation. Walkers track state transitions and maintain parsing history.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from copy import copy as shallow_copy
from typing import Any, Iterable, List, Optional, Self, Set

from lexpy import DAWG

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.types import StateType, VisitedEdgeType

logger = logging.getLogger(__name__)


class Walker(ABC):
    """Base class for state machine walkers.

    A `Walker` represents a position in a state machine graph and manages transitions
    between states as input is consumed. It tracks the current state, transition
    history, and accumulated values during parsing or generation.

    The `Walker` is the core component that enables efficient traversal of state machines
    without backtracking by maintaining multiple parallel paths through the graph.

    Attributes:
        acceptor: The token acceptor instance for this walker.
        current_state: Current position in the state machine.
        target_state: Target state for transitions in progress.
        transition_walker: Walker handling current transition.
        accept_history: History of accepted states.
        explored_edges: Set of visited state transitions.
        consumed_character_count: Number of characters processed.
        remaining_input: Unprocessed input string.
        _accepts_remaining_input: Whether walker can accept more input.
    """

    def __init__(
        self,
        acceptor: TokenAcceptor,
        current_state: Optional[StateType] = None,
    ) -> None:
        """Initialize a new Walker with the given acceptor.

        Args:
            acceptor: The token acceptor instance.
            current_state: The current state in the state machine.
        """
        self.acceptor = acceptor
        self.current_state: StateType = current_state or acceptor.initial_state
        self.target_state: Optional[StateType] = None
        self.transition_walker: Optional[Walker] = None
        self.accept_history: List[Walker] = []
        self.explored_edges: Set[VisitedEdgeType] = set()
        self.consumed_character_count: int = 0
        self.remaining_input: Optional[str] = None
        self._accepts_remaining_input: bool = False

    # -------- Abstract Methods --------

    @abstractmethod
    def consume_token(self, token: str) -> Iterable[Walker]:
        """Advance the walker with the given input token.

        Args:
            token: The token to process.

        Yields:
            Updated walker instances after advancement.
        """
        pass

    @abstractmethod
    def can_accept_more_input(self) -> bool:
        """Indicate whether the walker can accept more input for the current state.

        Returns:
            True if the walker can accept more input; False otherwise.
        """
        pass

    @abstractmethod
    def accumulated_value(self) -> Any:
        """Retrieve the current value accumulated by the walker.

        Returns:
            The current accumulated value.
        """
        pass

    @abstractmethod
    def is_within_value(self) -> bool:
        """Determine if the walker is currently within a value.

        Returns:
            True if in a value; False otherwise.
        """
        pass

    @property
    def current_edge(self) -> VisitedEdgeType:
        """Return the current edge as a tuple.

        Returns:
            A tuple representing the current edge.
        """
        return (
            self.current_state,
            self.target_state if self.target_state is not None else "$",
            str(self.accumulated_value()),
        )

    def should_start_transition(self, token: str) -> bool:
        """Determine if a transition should start with the given input token.

        Args:
            token: The token to process.

        Returns:
            True if the transition should start; False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.should_start_transition(token)
        # By default, we can start a transition unless we've already explored this edge.
        return self.current_edge not in self.explored_edges

    def should_complete_transition(
        self,
        transition_value: Any,
        is_end_state: bool,
    ) -> bool:
        """Determine if the transition should complete with the given parameters.

        Args:
            transition_value: The value accumulated during the transition.
            is_end_state: Indicates if the target state is an end state.

        Returns:
            True if the transition should complete; False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.should_complete_transition(
                transition_value, is_end_state
            )
        # By default, we can complete the transition.
        return True

    def clone(self) -> Self:
        """Create a shallow copy of the walker, including its accept history.

        Returns:
            A new Walker instance that is a clone of the current one.
        """
        cloned_walker = shallow_copy(self)
        cloned_walker.accept_history = self.accept_history.copy()
        cloned_walker.explored_edges = self.explored_edges.copy()
        return cloned_walker

    def transition(self) -> Walker:
        """Advance the walker to the next state.

        Yields:
            The walker instances after the transition.
        """
        if not self.transition_walker or self.target_state is None:
            logger.debug("No transition to complete: %s", self)
            return self

        self.accept_history.append(self.transition_walker)
        self.explored_edges.add(self.current_edge)
        logger.debug("Explored edges: %s", self.explored_edges)

        self.current_state = self.target_state
        logger.debug(f"{self} transitioned to state {self.current_state}")

        if not self.can_accept_more_input():
            self.transition_walker = None
            self.target_state = None

        if self.current_state in self.acceptor.end_states:
            from pse.state_machine.accepted_state import AcceptedState
            logger.debug("Walker in accepted state")
            return AcceptedState(self)

        return self

    def accepts_any_token(self) -> bool:
        """Check if the acceptor accepts any token (i.e., free text).

        Returns:
            True if all tokens are accepted; False otherwise.
        """
        return False

    def get_valid_continuations(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        """Return the set of strings that allow valid continuation from current state.

        The walker uses these strings to determine valid paths forward in the state
        machine, acting as an explorer probe to find acceptable transitions.

        Args:
            dawg: The Directed Acyclic Word Graph containing valid strings.
            depth: The current depth in the state machine traversal.

        Returns:
            A set of strings that represent valid continuations from current state.
        """
        logger.debug(
            "Getting valid continuations for walker:\n"
            "  Current State: %s\n"
            "  Depth: %d\n"
            "  DAWG Size: %d",
            self.current_state,
            depth,
            len(dawg),
        )
        return set()

    def find_valid_prefixes(self, dawg: DAWG) -> Set[str]:
        """Identify complete tokens that can advance the acceptor to a valid state.

        Args:
            dawg: The Directed Acyclic Word Graph to search for valid prefixes.

        Returns:
            A set of valid prefixes that can be used to advance the walker.
        """
        valid_prefixes = set()

        for continuation in self.get_valid_continuations(dawg):
            logger.debug("Getting tokens with prefix: %r", continuation)
            tokens = dawg.search_with_prefix(continuation)
            valid_prefixes.update(token for token in tokens if isinstance(token, str))

        logger.debug("Found %d valid tokens", len(valid_prefixes))
        return valid_prefixes

    def has_reached_accept_state(self) -> bool:
        """Check if the walker has reached an accepted (final) state.

        Returns:
            True if in an accepted state; False otherwise.
        """
        return False

    def _parse_value(self, value: Any) -> Any:
        """Parse the given value into an appropriate Python data type.

        Attempts to interpret the input value as a float, int, JSON object,
        or returns it as-is if none of the conversions are successful.

        Args:
            value: The value to parse.

        Returns:
            The parsed value as a float, int, dict, list, or the original value.
        """
        if not isinstance(value, str):
            return value

        # Try to parse as a float
        try:
            float_value = float(value)
            # Check if the float is actually an integer
            if float_value.is_integer():
                return int(float_value)
            return float_value
        except ValueError:
            pass

        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Return the original string if all parsing attempts fail
        return value

    # -------- Magic Methods --------

    def __hash__(self) -> int:
        """Generate a hash based on the walker's state and value.

        Returns:
            An integer hash value.
        """
        return hash(
            (
                self.current_state,
                self.target_state,
                str(self.accumulated_value()),
            )
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on the walker's state and accumulated value.

        Args:
            other: The object to compare with.

        Returns:
            True if both walkers are equal; False otherwise.
        """
        if not isinstance(other, Walker):
            return NotImplemented
        return (
            self.current_state == other.current_state
            and self.target_state == other.target_state
            and self.accumulated_value() == other.accumulated_value()
        )

    def __repr__(self) -> str:
        """Return a structured string representation of the walker.

        This method provides a detailed view of the walker's current state,
        transitions, and history, formatted for clarity and debugging.

        Returns:
            A formatted string showing state transitions, accumulated values,
            and other relevant walker details.
        """
        def _format_state_info() -> str:
            if self.current_state == self.acceptor.initial_state and not self.target_state:
                return ""
            state_info = f"State: {self.current_state}"
            return f"{state_info} ➔ {self.target_state}" if self.target_state else state_info

        def _format_history_info() -> str:
            if not self.accept_history:
                return ""
            history_values = [
                repr(w.accumulated_value())
                for w in self.accept_history
                if w.accumulated_value() is not None
            ]
            return f"History: {', '.join(history_values)}" if history_values else ""

        def _format_remaining_input() -> str:
            return f"Remaining input: `{self.remaining_input}`" if self.remaining_input else ""

        def _format_edge_info() -> str:
            if self.explored_edges:
                return _format_explored_edges()
            if self.target_state:
                return _format_current_edge()
            return ""

        def _format_explored_edges() -> str:
            edge_lines = []
            for from_state, to_state, edge_value in self.explored_edges:
                to_state_display = "End" if to_state == "$" else to_state
                edge_line = f"({from_state}) --[{repr(edge_value)}]--> ({to_state_display})"
                edge_lines.append(edge_line)
            return f"Explored edges: {', '.join(edge_lines)}"

        def _format_current_edge() -> str:
            to_state_display = "End" if self.target_state == "$" else self.target_state
            accumulated_value = repr(self.accumulated_value() or "")
            return f"Current edge: ({self.current_state}) --[{accumulated_value}]--> ({to_state_display})"

        def _format_transition_info() -> str:
            if not self.transition_walker:
                return ""
            transition_repr = repr(self.transition_walker)
            if '\n' not in transition_repr and len(transition_repr) < 40:
                return f"Transition: {transition_repr}"
            return f"Transition:\n  {transition_repr.replace('\n', '\n  ')}"

        # Build header with status indicators
        prefix = "✅ " if self.has_reached_accept_state() else ""
        suffix = " 🔄" if self.can_accept_more_input() else ""
        header = f"{prefix}{self.__class__.__name__}{suffix}"

        # Collect all information parts
        info_parts = [
            part for part in [
                _format_state_info(),
                _format_history_info(),
                _format_remaining_input(),
                _format_edge_info(),
                _format_transition_info()
            ] if part
        ]

        # Format final output
        # Format single line output if it fits within 80 chars
        single_line = (
            f"{header} {{{' | '.join(info_parts)}}}"
            if info_parts
            else f"{header} ({self.accumulated_value() or ''})"
        )
        if len(single_line) <= 80:
            return single_line

        # Format multiline output
        indent = '  '
        multiline_parts = [f"{header} {{"]
        for part in info_parts:
            if '\n' in part:
                part_lines = part.split('\n')
                multiline_parts.extend([indent + line for line in part_lines])
            else:
                multiline_parts.append(indent + part)
        multiline_parts.append('}')
        return '\n'.join(multiline_parts)
