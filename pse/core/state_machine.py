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
from typing import Iterable, Optional, Set, Tuple, Type

from lexpy import DAWG

from pse.acceptors.basic.acceptor import Acceptor
from pse.util.state_machine.types import Edge, State
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
        if not self.state_graph:
            yield initial_walker
        else:
            yield from self.branch_walker(initial_walker)

    def get_transition_walkers(
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
        logger.debug("🟡 Getting edges from state %s", current_state)
        for acceptor, target_state in self.get_edges(current_state):
            # create transition walkers from the edge's token acceptor
            for transition_walker in acceptor.get_walkers():
                yield transition_walker, current_state, target_state

            if (
                acceptor.is_optional
                and target_state not in self.end_states
                and walker.can_accept_more_input()
            ):
                logger.debug(
                    "🟢 %s supports pass-through to state %s", acceptor, target_state
                )
                yield from self.get_transition_walkers(walker, target_state)

    def advance(self, walker: Walker, token: str) -> Iterable[Walker]:
        """Advance the walker state with the given input token.

        Processes the input token and manages state transitions:
        1. If no active transition, branches to next possible states.
        2. If an active transition walker can proceed, advances it.
        3. If no valid transitions and the walker can accept more input, yields the walker.
        4. If the walker is optional and can accept more input, yields the walker.

        Args:
            walker: The walker to advance.
            token: The input string to process.

        Yields:
            Updated walkers after processing the token.
        """
        queue: deque[Tuple[Walker, str]] = deque([(walker, token)])

        while queue:
            current_walker, current_token = queue.popleft()

            if not current_walker.transition_walker:
                logger.debug(
                    "🟡 No Transition Walker for %s", current_walker.__class__.__name__
                )
                next_walkers = list(self.branch_walker(current_walker, current_token))

                if next_walkers:
                    new_walkers = [
                        (next_walker, current_token) for next_walker in next_walkers
                    ]
                    queue.extend(new_walkers)
                elif current_walker.remaining_input:
                    logger.debug(
                        "🟠 Yielding walker with remaining input: %s", current_walker
                    )
                    yield current_walker
                else:
                    logger.debug("🚫 No valid branches for %s", current_walker)

                continue

            if current_walker.should_start_transition(current_token):
                logger.debug(
                    "⚪️ Advancing from %s with token %s via %s",
                    current_walker.__class__.__name__,
                    repr(current_token),
                    current_walker.transition_walker.__class__.__name__,
                )
                for transition_walker in current_walker.transition_walker.consume_token(
                    current_token
                ):
                    if new_walker := current_walker.transition(transition_walker):
                        if new_walker.remaining_input:
                            logger.debug(
                                "⚪️ Walker with remaining input: %s", repr(new_walker)
                            )
                            queue.append((new_walker, new_walker.remaining_input))
                        else:
                            yield new_walker
                continue

            logger.debug(
                "🔴 %s cannot start transition with %s",
                current_walker.transition_walker.acceptor,
                repr(current_token),
            )

            if current_walker.transition_walker.can_accept_more_input():
                branched = False
                for next_transition_walker in current_walker.transition_walker.branch():
                    if next_transition_walker.should_start_transition(current_token):
                        current_walker.transition_walker = next_transition_walker
                        queue.append((current_walker, current_token))
                        branched = True
                if branched:
                    continue

            if current_walker.transition_walker.has_reached_accept_state():
                for next_walker in self.branch_walker(current_walker):
                    if next_walker.should_start_transition(current_token):
                        queue.append((next_walker, current_token))
                continue

            if current_walker.remaining_input:
                logger.debug(
                    "🟠 Yielding walker with remaining input: %s",
                    current_walker,
                )
                yield current_walker

    def branch_walker(
        self, walker: Walker, token: Optional[str] = None
    ) -> Iterable[Walker]:
        """Branch the walker into multiple paths for parallel exploration.

        At each decision point, clone the current walker for each possible transition.
        This allows simultaneous exploration of all valid paths without explicit backtracking.

        Args:
            walker: The current walker instance to branch from.
            token: Optional token to start transitions.

        Yields:
            New walker instances, each representing a different path.
        """
        logger.debug("🔵 Branching %s", repr(walker))
        for transition, start_state, target_state in self.get_transition_walkers(
            walker
        ):
            input_token = token or walker.remaining_input

            if input_token and not transition.should_start_transition(input_token):
                logger.debug(
                    "🔴 %s in %s cannot start with %s",
                    transition,
                    walker.acceptor,
                    repr(input_token),
                )
                continue

            if (
                walker.target_state == target_state
                and walker.transition_walker
                and walker.transition_walker.can_accept_more_input()
            ):
                logger.debug(
                    "🟡 Already exploring state %s, skipping %s",
                    target_state,
                    transition.acceptor,
                )
                continue

            logger.debug("🟢 Exploring %s to state %s", transition, target_state)
            yield walker.configure(transition, start_state, target_state)

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

        #
        # TODO: PARALLELIZE THIS FOR CHEAP
        #

        for walker in walkers:
            logger.debug("⚪️ Processing walker with token: %s", repr(token))
            for advanced_walker in walker.consume_token(token):
                # Full match
                if not advanced_walker.remaining_input:
                    logger.debug("🟢 Full match for token: %s", repr(token))
                    yield token, advanced_walker
                    continue

                if vocab is None:
                    logger.debug(
                        "🔴 No vocab provided, unable to check for partial match"
                    )
                    continue

                # Extract the valid prefix by removing remaining input
                prefix = token[: -len(advanced_walker.remaining_input)]
                if prefix and prefix in vocab:
                    logger.debug("🟢 Valid partial match: %s", repr(prefix))
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
        if self.transition_walker:
            if self.transition_walker.can_accept_more_input():
                return True

        return (
            bool(self.acceptor.state_graph.get(self.current_state))
            or self._accepts_more_input
        )

    def is_within_value(self) -> bool:
        """Determine if the walker is currently within a value.

        Returns:
            True if in a value, False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.is_within_value()

        return self.consumed_character_count > 0

    def accepts_any_token(self) -> bool:
        """Check if current transition matches all characters.

        Returns:
            True if matches all, False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.accepts_any_token()

        return False

    def consume_token(self, token: str) -> Iterable[Walker]:
        """Advance walker with input token.

        Args:
            token: Input to process.

        Yields:
            Updated walkers after advancement.
        """
        yield from self.acceptor.advance(self, token)

    def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        """Select valid characters based on current state.

        Args:
            dawg: DAWG to select from.
            depth: Current depth in state machine.

        Returns:
            Set of valid characters.
        """
        valid_prefixes = set()
        if self.transition_walker:
            valid_prefixes = self.transition_walker.get_valid_continuations(
                dawg, depth + 1
            )

        edge_walker_prefixes = set()
        if not edge_walker_prefixes:
            return valid_prefixes

        return valid_prefixes.union(edge_walker_prefixes)
