"""
A hierarchical state machine implementation for token-based parsing and validation.

This module provides a flexible state machine framework that:
- Supports parallel recursive descent parsing
- Enables efficient graph-based token acceptance
- Handles branching and backtracking through parallel walker exploration
- Allows composition of sub-state machines for complex grammars
- Provides case-sensitive and case-insensitive matching options

The core StateMachine class manages state transitions and token acceptance based on a
predefined graph structure, while the StateMachineWalker handles traversal and maintains
parsing state.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable, Optional, Tuple, Type

from lexpy import DAWG

from pse.acceptors.basic.acceptor import Acceptor
from pse.util.state_machine.types import Edge, State
from pse.util.state_machine.accepted_state import AcceptedState
from pse.core.walker import Walker

logger = logging.getLogger(__name__)


class StateMachine(Acceptor):
    """
    Non-Deterministic Hierarchical State Machine for managing token acceptance based on a predefined state graph.
    Includes support for optional acceptors and pass-through transitions, and parallel parsing.
    """

    @property
    def walker_class(self) -> Type[Walker]:
        """The walker class for this state machine."""
        return StateMachineWalker

    def get_edges(self, state: State) -> Iterable[Edge]:
        """Retrieve outgoing transitions for a given state.

        Args:
            state: The source state to get transitions from.

        Returns:
            An iterable of (acceptor, target_state) tuples representing possible transitions.
        """
        return self.state_graph.get(state, [])

    def get_walkers(self, state: Optional[State] = None) -> Iterable[Walker]:
        """Initialize walkers at the specified start state.

        If no graph is provided, only the initial walker is yielded.

        Args:
            state: The starting state. If None, uses the initial state.

        Yields:
            Walker instances positioned at the starting state.
        """
        initial_walker = self.walker_class(self, state)
        if self.state_graph:
            yield from self.branch_walker(initial_walker)
        else:
            yield initial_walker

    def get_transitions(
        self, walker: Walker, state: Optional[State] = None
    ) -> Iterable[Tuple[Walker, State, State]]:
        """Retrieve transition walkers from the current state.

        For each edge from the current state, yields walkers that can traverse that edge.
        Handles optional acceptors and pass-through transitions appropriately.

        Args:
            walker: The walker initiating the transition.
            state: Optional starting state. If None, uses the walker's current state.

        Returns:
            Iterable of tuples (transition_walker, source_state, target_state).
        """
        current_state = state or walker.current_state
        for acceptor, target_state in self.get_edges(current_state):
            for transition in acceptor.get_walkers():
                yield transition, current_state, target_state

            if (
                acceptor.is_optional
                and target_state not in self.end_states
                and walker.can_accept_more_input()
            ):
                logger.debug(f"🟢 {acceptor} supports pass-through to state {target_state}")
                yield from self.get_transitions(walker, target_state)

    def branch_walker(
        self, walker: Walker, token: Optional[str] = None
    ) -> Iterable[Walker]:
        """
        Branch the walker into multiple paths for parallel exploration.

        Args:
            walker: The current walker to branch from.
            token: Optional token to start transitions.

        Yields:
            New walker instances, each representing a different path.
        """
        logger.debug(f"🔵 Branching {repr(walker)}")
        input_token = token or walker.remaining_input

        for transition, start_state, target_state in self.get_transitions(walker):
            if branched_walker := walker.start_transition(
                transition,
                input_token,
                start_state,
                target_state,
            ):
                yield branched_walker
                continue

            if (
                transition.acceptor.is_optional
                and target_state in self.end_states
                and (walker.remaining_input or token)
            ):
                logger.debug(f"🟠 {transition} is optional; yielding accepted state")
                if not walker.remaining_input:
                    walker.remaining_input = token
                yield AcceptedState(walker)

    def advance(self, walker: Walker, input_token: str) -> Iterable[Walker]:
        """Process a token through the state machine, advancing walker states and managing transitions.

        This method implements a breadth-first traversal of possible state transitions, handling:
        1. Initial state branching when no transition is active
        2. Active transition advancement
        3. Transition branching when current path is blocked
        4. Accepted state processing
        5. Partial match yielding

        Args:
            walker: Current walker instance containing state and transition information
            input_token: Input string to process

        Yields:
            Walker: Updated walker instances representing:
                - Completed transitions
                - Partial matches
                - Valid branches
        """
        queue: deque[Tuple[Walker, str]] = deque([(walker, input_token)])

        def handle_blocked_transition(blocked_walker: Walker, token: str) -> Iterable[Walker]:
            """Handle blocked transitions."""
            branched_walkers = []
            for branched_walker in blocked_walker.branch(token):
                if branched_walker.should_start_transition(token):
                    branched_walkers.append(branched_walker)
                elif branched_walker.has_reached_accept_state():
                    logger.debug(f"🟠 Walker has reached accept state: {repr(branched_walker)}")
                    yield branched_walker
                    return

            queue.extend((new_walker, token) for new_walker in branched_walkers)
            if not branched_walkers and blocked_walker.remaining_input:
                logger.debug(f"🟠 Walker has remaining input: {repr(blocked_walker)}")
                yield blocked_walker
            elif not branched_walkers:
                logger.debug(f"🔴 {repr(blocked_walker)} cannot parse {repr(token)}.")

        while queue:
            current_walker, current_token = queue.popleft()

            # Handle case where transition cannot be started
            if (
                not current_walker.transition_walker or
                not current_walker.should_start_transition(current_token)
            ):
                yield from handle_blocked_transition(current_walker, current_token)
                continue

            # Handle active transition
            logger.debug(f"⚪️ Parsing {repr(current_token)} via {repr(current_walker)}")
            for transition in current_walker.transition_walker.consume_token(current_token):
                if new_walker := current_walker.complete_transition(transition):
                    if new_walker.remaining_input:
                        queue.append((new_walker, new_walker.remaining_input))
                    else:
                        yield new_walker

    @staticmethod
    def advance_all(
        walkers: Iterable[Walker], token: str, vocab: Optional[DAWG] = None
    ) -> Iterable[Tuple[str, Walker]]:
        """Advance all walkers in parallel to find valid token matches.

        Processes multiple walkers concurrently to find valid token matches and partial matches.
        Uses a thread pool to parallelize walker advancement for better performance.

        Args:
            walkers: Collection of walker instances to advance in parallel
            token: Input token string to match against
            vocab: Optional DAWG vocabulary to validate partial token matches.
                  If provided, enables partial matching by checking prefixes.

        Returns:
            An iterable of tuples containing:
            - str: The matched token or valid prefix
            - Walker: The advanced walker instance after consuming the token

        For each walker:
        1. Attempts to consume the input token
        2. If a full match is found (no remaining input), yields the match
        3. If partial match is found and vocab is provided, validates the prefix
           against the vocab and yields valid partial matches
        """
        for walker in walkers:
            logger.debug(f"⚪️ Processing walker with token: {token}")
            for advanced_walker in walker.consume_token(token):
                if not advanced_walker.remaining_input:
                    logger.debug(f"🟢 Full match for token: {repr(token)}")
                    yield token, advanced_walker
                    continue

                if vocab is None:
                    logger.debug("🔴 No vocab - unable to check for partial match")
                    continue

                # Extract the valid prefix by removing remaining input
                prefix = token[: -len(advanced_walker.remaining_input)]
                if prefix and prefix in vocab:
                    logger.debug(f"🟢 Valid partial match: {repr(prefix)}")
                    advanced_walker.remaining_input = None
                    yield prefix, advanced_walker


class StateMachineWalker(Walker):
    """Walker for navigating through StateMachine states.

    Manages state traversal and tracks:
    - Current position
    - Transition state
    - Input processing
    - Value accumulation
    """

    def can_accept_more_input(self) -> bool:
        """Check if walker can process more input.

        Returns:
            True if more input can be handled, False otherwise.
        """
        if self.transition_walker and self.transition_walker.can_accept_more_input():
            return True

        return self._accepts_more_input or bool(self.acceptor.state_graph.get(self.current_state))

    def accepts_any_token(self) -> bool:
        """Check if current transition matches all characters.

        Returns:
            True if matches all, False otherwise.
        """

        return (
            self.transition_walker.accepts_any_token()
            if self.transition_walker
            else False
        )

    def is_within_value(self) -> bool:
        """Determine if the walker is currently within a value.

        Returns:
            True if in a value, False otherwise.
        """
        return (
            self.transition_walker.is_within_value()
            if self.transition_walker
            else self.consumed_character_count > 0
        )

    def consume_token(self, token: str) -> Iterable[Walker]:
        """Advance walker with input token.

        Args:
            token: Input to process.

        Yields:
            Updated walkers after advancement.
        """
        yield from self.acceptor.advance(self, token)
