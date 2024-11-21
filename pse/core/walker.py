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

from pse.acceptors.basic.acceptor import Acceptor
from pse.util.state_machine.types import State, VisitedEdge

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
        accepted_history: History of accepted states.
        explored_edges: Set of visited state transitions.
        consumed_character_count: Number of characters processed.
        remaining_input: Unprocessed input string.
        _accepts_more_input: Whether walker can accept more input.
    """

    def __init__(
        self,
        acceptor: Acceptor,
        current_state: Optional[State] = None,
    ) -> None:
        """Initialize a new Walker with the given acceptor.

        Args:
            acceptor: The token acceptor instance.
            current_state: The current state in the state machine.
        """
        self.acceptor = acceptor
        self.accepted_history: List[Walker] = []
        self.explored_edges: Set[VisitedEdge] = set()

        self.current_state: State = current_state or acceptor.start_state
        self.target_state: Optional[State] = None
        self.transition_walker: Optional[Walker] = None

        self.consumed_character_count: int = 0
        self.remaining_input: Optional[str] = None

        self._raw_value: Optional[str] = None
        self._accepts_more_input: bool = False

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
    def is_within_value(self) -> bool:
        """Determine if the walker is currently within a value.

        Returns:
            True if in a value; False otherwise.
        """
        pass

    @property
    def current_value(self) -> Any:
        """Retrieve the accumulated walker value.

        Returns:
            The current value from transition or history, parsed into appropriate type.
            Returns None if no value is accumulated.
        """
        return self._parse_value(self.raw_value) if self.raw_value else None

    @property
    def raw_value(self) -> Optional[str]:
        """Retrieve the raw accumulated value as a string.

        Returns:
            The concatenated raw values from history and transitions.
        """
        if self._raw_value:
            return self._raw_value

        if not self.accepted_history and not self.transition_walker:
            return None

        history = [walker.raw_value for walker in self.accepted_history]
        if self.transition_walker:
            history.append(self.transition_walker.raw_value)

        return "".join(value for value in history if value is not None)

    @property
    def current_edge(self) -> VisitedEdge:
        """Return the current edge as a tuple.

        Returns:
            A tuple representing the current edge.
        """
        return (
            self.current_state,
            self.target_state,
            self.raw_value,
        )

    def clone(self) -> Self:
        """Creates a shallow copy of the walker with copied history and explored edges."""
        cloned_walker = shallow_copy(self)
        cloned_walker.accepted_history = self.accepted_history.copy()
        cloned_walker.explored_edges = self.explored_edges.copy()
        return cloned_walker

    def should_start_transition(self, token: str) -> bool:
        """Determine if a transition should start with the given input token.

        Args:
            token: The token to process.

        Returns:
            True if the transition should start; False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.should_start_transition(token)

        if self.current_edge in self.explored_edges:
            self._accepts_more_input = False
            return False

        return True

    def should_complete_transition(self) -> bool:
        """Determine if the transition should complete with the given parameters.

        Args:
            transition_value: The value accumulated during the transition.
            is_end_state: Indicates if the target state is an end state.

        Returns:
            True if the transition should complete; False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.should_complete_transition()

        return True

    def start_transition(
        self,
        transition_walker: Walker,
        token: Optional[str] = None,
        start_state: Optional[State] = None,
        target_state: Optional[State] = None,
    ) -> Optional[Walker]:
        """Start a new transition with the given token."""

        if (
            token is not None
            and not transition_walker.should_start_transition(token)
        ):
            return None

        if (
            self.target_state == target_state
            and self.transition_walker
            and self.transition_walker.can_accept_more_input()
        ):
            return None

        clone = self.clone()
        clone.current_state = start_state or clone.current_state
        clone.target_state = target_state

        if (
            clone.transition_walker is not None
            and clone.transition_walker.has_reached_accept_state()
        ):
            clone.accepted_history.append(clone.transition_walker)

        clone.transition_walker = transition_walker

        return clone

    def complete_transition(
        self,
        transition_walker: Walker,
    ) -> Optional[Walker]:
        """Advance the walker to the next state.

        Args:
            transition_walker: The walker handling the current transition.

        Returns:
            A new Walker instance after the transition.
        """
        clone = self.clone()
        clone.transition_walker = transition_walker

        clone.remaining_input = transition_walker.remaining_input
        clone.transition_walker.remaining_input = None

        clone.consumed_character_count += transition_walker.consumed_character_count
        clone.explored_edges.add(clone.current_edge)

        if not clone.should_complete_transition():
            return clone if clone.can_accept_more_input() else None

        if (
            clone.target_state is not None
            and transition_walker.has_reached_accept_state()
        ):
            clone.current_state = clone.target_state

            if not transition_walker.can_accept_more_input():
                clone.accepted_history.append(transition_walker)
                clone.transition_walker = None
                clone.target_state = None

            if clone.current_state in clone.acceptor.end_states:
                from pse.util.state_machine.accepted_state import AcceptedState

                return AcceptedState(clone)

        return clone

    def branch(self, token: Optional[str] = None) -> Iterable[Walker]:
        """Branch the current walker or transition walker to explore all possible
        transitions from the current state.

        Yields:
            New walker instances representing different paths.
        """
        if self.transition_walker:
            transition_walkers = []
            if self.transition_walker.can_accept_more_input():
                transition_walkers.extend(self.transition_walker.branch(token))

            for new_transition_walker in transition_walkers:
                clone = self.clone()
                clone.transition_walker = new_transition_walker
                yield clone

            if (
                not transition_walkers
                and not self.transition_walker.has_reached_accept_state()
            ):
                return

        yield from self.acceptor.branch_walker(self, token)

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
        if not self.transition_walker or depth > 10:
            return set()

        return self.transition_walker.get_valid_continuations(dawg, depth + 1)

    def find_valid_prefixes(self, dawg: DAWG) -> Set[str]:
        """Identify complete tokens that can advance the acceptor to a valid state.

        Args:
            dawg: The Directed Acyclic Word Graph to search for valid prefixes.

        Returns:
            A set of valid prefixes that can be used to advance the walker.
        """

        valid_prefixes: Set[str] = set()

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

    # -------- Helper Methods --------

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
                return int(value)
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

    def _format_explored_edges(self) -> str:
        """Format the explored edges for logging purposes.

        Returns:
            A string representation of the explored edges.
        """
        edge_lines = []
        for from_state, to_state, edge_value in sorted(
            self.explored_edges,
            key=lambda x: (
                x[2],
                str(x[0]) if x[0] != "$" else "End",
                str(x[1]) if x[1] != "$" else "End",
            ),
        ):
            to_state_display = "End" if to_state == "End" else to_state
            edge_line = f"({from_state}) --{repr(edge_value)}--> ({to_state_display})"
            edge_lines.append(edge_line)
        return f"Explored edges:\n  {'\n  '.join(edge_lines)}"

    # -------- Dunder Methods --------

    def __hash__(self) -> int:
        """Generate a hash based on the walker's state and value.

        Returns:
            An integer hash value.
        """
        return hash(
            (
                self.current_state,
                self.target_state,
                self.raw_value,
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
            and self.current_value == other.current_value
        )

    def __str__(self) -> str:
        if self.transition_walker:
            return f"{self.acceptor}.Walker({self.transition_walker})"
        return self.__repr__()

    def __repr__(self) -> str:
        """Return a structured string representation of the walker.

        This method provides a detailed view of the walker's current state,
        transitions, and history, formatted for clarity and debugging.

        Returns:
            A formatted string showing state transitions, accumulated values,
            and other relevant walker details.
        """

        def _format_state_info() -> str:
            state_info = f"State: {self.current_state}"
            return (
                f"{state_info} ➔ {self.target_state}"
                if self.target_state is not None
                and self.current_state != self.target_state
                else state_info
            )

        def _format_history_info() -> str:
            if not self.accepted_history:
                return ""
            history_values = [
                repr(w.current_value)
                for w in self.accepted_history
                if w.current_value is not None
            ]
            return f"History: {', '.join(history_values)}" if history_values else ""

        def _format_remaining_input() -> str:
            return (
                f"Remaining input: {repr(self.remaining_input)}"
                if self.remaining_input
                else ""
            )

        def _format_current_edge() -> str:
            to_state_display = self.target_state or "?"
            accumulated_value = self.raw_value or self.current_value or ""
            return (
                f"Current edge: ({self.current_state}) "
                f"--{repr(accumulated_value)}--> ({to_state_display})"
            )

        def _format_transition_info() -> str:
            if not self.transition_walker:
                return ""
            transition_repr = repr(self.transition_walker)
            if "\n" not in transition_repr and len(transition_repr) < 40:
                return f"Transition: {transition_repr}"
            return f"Transition:\n  {transition_repr.replace('\n', '\n  ')}"

        # Build header with status indicators
        prefix = "✅ " if self.has_reached_accept_state() else ""
        suffix = " 🔄" if self._accepts_more_input else ""
        header = f"{prefix}{self.acceptor.__class__.__name__}.Walker{suffix}"

        # Collect all information parts
        info_parts = [
            part
            for part in [
                _format_state_info(),
                _format_history_info(),
                _format_current_edge(),
                _format_remaining_input(),
                _format_transition_info(),
            ]
            if part
        ]

        # Format final output
        # Format single line output if it fits within 80 chars
        single_line = (
            f"{header} ({', '.join(info_parts)})" if info_parts else f"{header}()"
        )
        if len(single_line) <= 80:
            return single_line

        # Format multiline output
        indent = "  "
        multiline_parts = [f"{header} {{"]
        for part in info_parts:
            if "\n" in part:
                part_lines = part.split("\n")
                multiline_parts.extend([indent + line for line in part_lines])
            else:
                multiline_parts.append(indent + part)

        multiline_parts.append("}")
        return "\n".join(multiline_parts)
