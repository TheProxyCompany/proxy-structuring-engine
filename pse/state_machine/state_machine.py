from __future__ import annotations

import json
import logging
from pprint import pformat
from typing import Any, Iterable, List, Optional, Set, Tuple, Type

from lexpy import DAWG

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.cursor import Cursor, MAX_DEPTH
from pse.state_machine.types import EdgeType, StateMachineGraph, StateType

logger = logging.getLogger(__name__)


class StateMachine(TokenAcceptor):
    """A token acceptor that operates based on a state graph defining transitions between states.

    Each state can have multiple edges, defined by a target state and a TokenAcceptor.
    Upon reaching an accepted state, a transition to the target state is triggered.
    This process repeats until the state machine reaches a final state.
    Multiple transition paths are explored in parallel to efficiently handle various input possibilities.
    """

    def __init__(
        self,
        graph: Optional[StateMachineGraph] = None,
        initial_state: StateType = 0,
        end_states: Optional[Iterable[StateType]] = None,
    ) -> None:
        """Initialize the StateMachine with a state graph.

        Args:
            graph: A dictionary mapping each state to a list of tuples,
                each containing a TokenAcceptor and the target state.
                Defaults to an empty dictionary if not provided.
            initial_state: The starting state of the state machine. Defaults to 0.
            end_states: A collection of states considered final or accepting.
                Defaults to ["$"] if not provided.
        """
        super().__init__(initial_state, end_states or ["$"])
        self.graph: StateMachineGraph = graph or {}

    @property
    def cursor_class(self) -> Type[Cursor]:
        """Return the cursor class associated with this StateMachine.

        Returns:
            The class of the cursor.
        """
        return self.__class__.Cursor

    def get_edges(self, state: StateType) -> List[EdgeType]:
        """Retrieve outgoing edges (transitions) for a given state.

        Args:
            state: The state from which to retrieve outgoing edges.

        Returns:
            A list of edges represented as tuples containing a TokenAcceptor and the target state.
        """
        return self.graph.get(state, [])

    def get_cursors(self) -> Iterable[Cursor]:
        """Initialize and retrieve cursors starting from the initial state.

        Returns:
            An iterable of cursors initialized at the starting state.
        """
        initial_cursor = self.cursor_class(self)
        initial_cursor.current_state = self.initial_state
        return self._find_transitions(initial_cursor, [], set())

    def advance_cursor(self, cursor: Cursor, input_str: str) -> Iterable[Cursor]:
        """Advance a cursor with the given input and handle state transitions.

        Args:
            cursor: The cursor to advance.
            input_str: The input string to process.

        Yields:
            Updated cursors after advancement.
        """
        if not cursor.transition_cursor:
            if not self.expects_more_input(cursor):
                return
            yield cursor
            return

        logger.debug(
            "\nAdvancing cursor:\n"
            f"  Cursor: {pformat(cursor, indent=2)}\n"
            f"  Input: '{input_str}'"
        )
        successfully_advanced = False
        for followup_cursor in cursor.transition_cursor.advance(input_str):
            successfully_advanced = True
            advanced_cursor = cursor.clone()
            advanced_cursor.transition_cursor = followup_cursor
            advanced_cursor.remaining_input = followup_cursor.remaining_input
            advanced_cursor.consumed_character_count += followup_cursor.consumed_character_count
            advanced_cursor._accepts_remaining_input = followup_cursor.can_handle_remaining_input

            if self._should_skip_cursor(advanced_cursor):
                continue

            if not followup_cursor.in_accepted_state() and not followup_cursor.remaining_input:
                yield advanced_cursor
                continue

            for new_cursor in self._cascade_transition(advanced_cursor, [], set()):
                if advanced_cursor.remaining_input and self.expects_more_input(new_cursor):
                    yield from self.advance_cursor(new_cursor, advanced_cursor.remaining_input)
                else:
                    yield new_cursor

        logger.debug(
            "\nAdvancement status:\n"
            f"  Successfully advanced: {successfully_advanced}\n"
            f"  Cursor: {pformat(cursor, indent=2)}\n"
            f"  Can handle remaining input: {cursor.can_handle_remaining_input}"
        )

        if not successfully_advanced and cursor.can_handle_remaining_input:
            cursor._accepts_remaining_input = False
            if cursor.accept_history and cursor.accept_history[-1].can_handle_remaining_input:
                transition_cursor = cursor.accept_history[-1].clone()
                cursor.accept_history.pop()
                cursor.transition_cursor = transition_cursor
                cursor.target_state = cursor.current_state

            yield from cursor.advance(input_str)

    def _find_transitions(
        self,
        cursor: Cursor,
        visited_states: List[Tuple[StateType, Any]],
        traversed_edges: Set[Tuple[StateType, Any]],
    ) -> Iterable[Cursor]:
        """Recursively find and create transitions for the given cursor.

        Args:
            cursor: The current cursor to process.
            visited_states: List of states already visited to avoid cycles.
            traversed_edges: Set of traversed edges to prevent duplication.

        Yields:
            New cursors after processing transitions.
        """
        edges = self.get_edges(cursor.current_state)

        if not edges and not cursor.in_accepted_state() and cursor.remaining_input:
            yield cursor

        for acceptor, target_state in edges:
            if cursor.start_transition(acceptor, target_state):
                for transition_cursor in acceptor.get_cursors():
                    new_cursor = self._create_transition_cursor(cursor, transition_cursor, target_state)
                    if transition_cursor.in_accepted_state() and not acceptor.expects_more_input(transition_cursor):
                        state_snapshot = (new_cursor.current_state, str(new_cursor.get_value()))
                        if state_snapshot in visited_states:
                            continue

                        new_visited_states = visited_states + [state_snapshot]
                        yield from self._cascade_transition(new_cursor, new_visited_states, traversed_edges)
                    else:
                        yield new_cursor

    def _cascade_transition(
        self,
        cursor: Cursor,
        visited_states: List[Tuple[StateType, Any]],
        traversed_edges: Set[Tuple[StateType, Any]],
    ) -> Iterable[Cursor]:
        """Handle transitions that reach an accepted state and cascade to the next state.

        Args:
            cursor: The cursor that reached an accepted state.
            visited_states: List of states already visited to avoid cycles.
            traversed_edges: Set of traversed edges to prevent duplication.

        Yields:
            Cursors after cascading the transition.
        """
        if not cursor.transition_cursor or cursor.target_state is None:
            raise AssertionError("Cursor must have a transition_cursor and target_state.")

        transition_value = cursor.transition_cursor.get_value()
        target_state = cursor.target_state
        is_end_state = target_state in self.end_states

        if cursor.complete_transition(transition_value, target_state, is_end_state):
            cursor.accept_history.append(cursor.transition_cursor)
            cursor.current_state = target_state
            cursor.target_state = None
            cursor.transition_cursor = None

            edge = (cursor.current_state, str(cursor.get_value()))

            if edge not in traversed_edges:
                traversed_edges.add(edge)
                if is_end_state and not cursor.remaining_input:
                    yield AcceptedState(cursor)

                yield from self._find_transitions(cursor, visited_states, traversed_edges)

    def _create_transition_cursor(
        self, cursor: Cursor, transition_cursor: Cursor, target_state: StateType
    ) -> Cursor:
        """Create a new cursor for state transitions.

        Args:
            cursor: The current cursor.
            transition_cursor: The cursor handling the transition.
            target_state: The state to transition to.

        Returns:
            A new cursor set up for the transition.
        """
        new_cursor = cursor.clone()
        new_cursor.transition_cursor = transition_cursor
        new_cursor.target_state = target_state
        return new_cursor

    def _should_skip_cursor(self, cursor: Cursor) -> bool:
        """Determine if the cursor should be skipped based on its ability to handle remaining input.

        Args:
            cursor: The cursor to evaluate.

        Returns:
            True if the cursor should be skipped, False otherwise.
        """
        if (
            not cursor.transition_cursor
            or not cursor.transition_cursor.remaining_input
            or cursor.transition_cursor.can_handle_remaining_input
        ):
            return False

        return not self.expects_more_input(cursor)

    def expects_more_input(self, cursor: Cursor) -> bool:
        """Determine if the state machine expects more input based on the current cursor state.

        Args:
            cursor: The current cursor.

        Returns:
            True if more input is expected, False otherwise.
        """
        if cursor.in_accepted_state() or cursor.current_state in self.end_states:
            return False

        if cursor.remaining_input:
            return True

        return bool(self.get_edges(cursor.current_state))

    class Cursor(Cursor):
        """Cursor for navigating through states in the StateMachine."""

        @property
        def can_handle_remaining_input(self) -> bool:
            """Determine if the cursor can handle more input.

            Returns:
                True if the cursor or its transition cursor can handle more input.
            """
            if not self._accepts_remaining_input:
                return False

            if self.transition_cursor:
                return self.transition_cursor.can_handle_remaining_input

            if self.accept_history:
                while (
                    self.accept_history
                    and isinstance(self.accept_history[-1], AcceptedState)
                    and self.accept_history[-1].is_empty_transition()
                ):
                    self.accept_history.pop()

                if self.accept_history:
                    return self.accept_history[-1].can_handle_remaining_input

            if isinstance(self.acceptor, StateMachine):
                return bool(self.acceptor.get_edges(self.current_state))

            return False

        def matches_all(self) -> bool:
            """Check if the current transition cursor matches all possible characters.

            Returns:
                True if it matches all, False otherwise.
            """
            if self.transition_cursor:
                return self.transition_cursor.matches_all()
            return False

        def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
            """Select valid characters based on the current transition cursor.

            Args:
                dawg: The DAWG to select from.
                depth: The current depth of the cursor in the state machine.

            Returns:
                A set of valid characters accepted by the transition cursor.
            """
            if depth >= MAX_DEPTH:
                return set()

            valid_prefixes = set()
            if self.transition_cursor:
                valid_prefixes = self.transition_cursor.select(dawg, depth + 1)

            edge_cursor_prefixes = set()
            depth_prefix = f"{depth * ' '}{depth}. " if depth > 0 else ""
            if self.target_state is not None and isinstance(self.acceptor, StateMachine) and self.in_accepted_state():
                for acceptor, state in self.acceptor.get_edges(self.target_state):
                    # Go through the possible cursors from each edge's acceptor
                    logger.debug(
                        "\n%sDownstream edge analysis:\n"
                        f"  Source state: {self.target_state}\n"
                        f"  Target state: {state}\n"
                        f"  Acceptor: {acceptor.__class__.__name__}\n"
                        f"  Acceptor details: {pformat(acceptor, indent=2)}",
                        depth_prefix
                    )
                    for possible_next_cursor in acceptor.get_cursors():
                        prefixes = possible_next_cursor.select(dawg, depth + 1)
                        edge_cursor_prefixes.update(prefixes)

            logger.debug(
                "\nSelecting prefixes at depth %d:\n"
                f"  Current state: {self.current_state}\n"
                f"  Target state: {self.target_state}\n"
                f"  Valid prefixes: {pformat(valid_prefixes, indent=2)}\n"
                f"  Edge cursor prefixes: {pformat(edge_cursor_prefixes, indent=2)}"
            )
            if not edge_cursor_prefixes:
                return valid_prefixes
            return valid_prefixes.union(edge_cursor_prefixes)

        def advance(self, input_str: str) -> Iterable[Cursor]:
            """Advance the cursor with the given input.

            Args:
                input_str: The input string to process.

            Yields:
                Updated cursors after advancement.
            """
            return self.acceptor.advance_cursor(self, input_str)

        def get_value(self) -> Any:
            """Retrieve the accumulated value from the cursor's history.

            Returns:
                The value from the transition_cursor if present and has a value,
                otherwise, the accumulated values from accept_history, or None if not available.
            """
            if self.transition_cursor:
                return self.transition_cursor.get_value()

            if self.accept_history:
                values = [cursor.get_value() for cursor in self.accept_history if cursor.get_value() is not None]
                if values:
                    concatenated = ''.join(map(str, values)) if len(values) > 1 else values[0]
                    return self._parse_value(concatenated)

            return None

        def _parse_value(self, value: Any) -> Any:
            """Parse the given value into an appropriate type.

            Args:
                value: The value to parse.

            Returns:
                The parsed value.
            """
            if not isinstance(value, str):
                return value

            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                return int(value)
            except ValueError:
                pass

            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

            return value

        def is_in_value(self) -> bool:
            """Determine if the cursor is currently within a value.

            Returns:
                True if in a value, False otherwise.
            """
            if self.consumed_character_count > 0 and self.transition_cursor:
                return self.transition_cursor.is_in_value()
            if self.accept_history:
                return self.accept_history[-1].is_in_value()
            return False

        def __hash__(self) -> int:
            """Compute the hash value of the cursor.

            Returns:
                The hash value.
            """
            return hash((self.current_state, self.target_state, str(self.get_value())))

        def __repr__(self) -> str:
            """Return the string representation of the cursor.

            Returns:
                The string representation.
            """
            components = []

            if self.accept_history:
                history = ''.join(str(cursor.get_value())
                                for cursor in self.accept_history
                                if cursor.get_value())
                if history:
                    components.append(f"history={repr(history)}")

            components.append(f"state={self.current_state}")

            if self.target_state is not None:
                components.append(f"target={self.target_state}")

            if self.transition_cursor:
                components.append(f"via={self.transition_cursor}")

            if self.remaining_input:
                components.append(f"remaining='{self.remaining_input}'")

            return f"{self.acceptor.__class__.__name__}.Cursor(\n  " + \
                   ",\n  ".join(components) + "\n)"


__all__ = ["StateMachine", "StateMachineGraph", "StateType"]
