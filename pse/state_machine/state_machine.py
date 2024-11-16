"""State machine implementation for token acceptance based on graph traversal."""

from __future__ import annotations

import logging
from collections import deque
from pprint import pformat
from typing import Iterable, Optional, Set, Tuple, Type

from lexpy import DAWG

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.types import EdgeType, StateMachineGraph, StateType
from pse.state_machine.walker import Walker

logger = logging.getLogger(__name__)


class StateMachine(TokenAcceptor):
    """StateMachine for managing token acceptance based on a predefined graph."""

    def __init__(
        self,
        graph: Optional[StateMachineGraph] = None,
        initial_state: StateType = 0,
        end_states: Optional[Iterable[StateType]] = None,
        walker_type: Optional[Type[Walker]] = None,
        is_optional: bool = False,
        is_case_sensitive: bool = True,
    ) -> None:
        """Initialize the StateMachine with a state graph.

        Args:
            graph: Mapping of states to lists of (TokenAcceptor, target_state) tuples.
                  Defaults to an empty dictionary.
            initial_state: The starting state. Defaults to 0.
            end_states: Collection of accepting states. Defaults to ("$",).
            walker_type: The type of walker to use. Defaults to `StateMachineWalker`.
            is_optional: Whether the acceptor is optional. Defaults to False.
            is_case_sensitive: Whether the acceptor is case sensitive. Defaults to True.
        """
        super().__init__(
            graph or {},
            initial_state,
            end_states or ["$"],
            walker_type or StateMachineWalker,
            is_optional,
            is_case_sensitive,
        )

    def get_edges(self, state: StateType) -> Iterable[EdgeType]:
        """Retrieve outgoing transitions for a given state.

        Args:
            state: The source state to get transitions from.

        Returns:
            An iterable of (acceptor, target_state) tuples representing possible transitions.
        """
        return self.graph.get(state, [])

    def get_walkers(self, state: Optional[StateType] = None) -> Iterable[Walker]:
        """Initialize walkers at the specified start state.

        If no graph is provided, only the initial walker is yielded.

        Args:
            state: The starting state. If None, uses the initial state.

        Yields:
            Walker instances positioned at the starting state.
        """
        initial_walker = self._walker(self, state)
        if not self.graph:
            yield initial_walker
        else:
            yield from self.branch_walker(initial_walker)

    def advance_walker(self, walker: Walker, token: str) -> Iterable[Walker]:
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
                logger.debug("🟡 No Transition Walker for %s", current_walker.__class__.__name__)
                next_walkers = list(self.branch_walker(current_walker, current_token))

                if next_walkers:
                    queue.extend((next_walker, current_token) for next_walker in next_walkers)
                elif current_walker.remaining_input:
                    logger.debug("🟠 Yielding walker with remaining input: %s", current_walker)
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
                for transition_walker in current_walker.transition_walker.consume_token(current_token):
                    if new_walker := current_walker.transition(transition_walker):
                        if new_walker.remaining_input:
                            logger.debug("⚪️ Walker with remaining input: %s", repr(new_walker))
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

    def get_transition_walkers(
        self, walker: Walker, state: Optional[StateType] = None
    ) -> Iterable[Tuple[Walker, StateType, StateType]]:
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
                logger.debug("🟢 %s supports pass-through to state %s", acceptor, target_state)
                yield from self.get_transition_walkers(walker, target_state)

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
        logger.debug("🔵 Branching %s", walker)
        for transition, start_state, target_state in self.get_transition_walkers(walker):
            input_token = token or walker.remaining_input

            if input_token and not transition.should_start_transition(input_token):
                logger.debug("🔴 %s in %s cannot start with %s", transition, walker.acceptor, repr(input_token))
                if transition.acceptor.is_optional:
                    logger.debug("🟢 %s supports pass-through to state %s", transition.acceptor, target_state)
                continue

            if walker.target_state == target_state and walker.transition_walker and walker.transition_walker.can_accept_more_input():
                logger.debug("🟡 Already exploring state %s, skipping %s", target_state, transition.acceptor)
                continue

            logger.debug("🟢 Exploring %s to state %s", transition, target_state)
            yield walker.configure(transition, start_state, target_state)

    @classmethod
    def advance_all_walkers(
        cls,
        walkers: Iterable[Walker],
        token: str,
        dawg: Optional[DAWG] = None,
    ) -> Iterable[Tuple[str, Walker]]:
        """Advance all walkers with the given input.

        Args:
            walkers: An iterable of walker instances.
            token: The input string to advance the walkers with.
            dawg: Optional DAWG to optimize prefix matching.

        Returns:
            An iterable of (advanced token, updated walker) tuples.
        """
        if not walkers:
            return []

        def process_walker(walker: Walker) -> Iterable[Tuple[str, Walker]]:
            logger.debug("⚪️ Processing walker with token: %s", repr(token))
            for advanced_walker in walker.consume_token(token):
                if not advanced_walker.remaining_input:
                    logger.debug("🟢 Full match for token: %s", repr(token))
                    yield token, advanced_walker
                elif (
                    dawg is not None
                    and (valid_prefix := token[:-len(advanced_walker.remaining_input)]) in dawg
                ):
                    logger.debug("🟢 Valid partial match: %s", repr(valid_prefix))
                    advanced_walker.remaining_input = None
                    yield valid_prefix, advanced_walker

        for result in cls._EXECUTOR.map(process_walker, walkers):
            yield from result


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
            bool(self.acceptor.graph.get(self.current_state))
            or self._accepts_more_input
        )

    def is_within_value(self) -> bool:
        """Determine if the walker is currently within a value.

        Returns:
            True if in a value, False otherwise.
        """
        if self.consumed_character_count > 0 and self.transition_walker:
            return self.transition_walker.is_within_value()

        return False

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
        yield from self.acceptor.advance_walker(self, token)

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
        depth_prefix = f"{depth * ' '}{depth}. " if depth > 0 else ""
        if (
            self.target_state is not None
            and isinstance(self.acceptor, StateMachine)
            and self.has_reached_accept_state()
        ):
            for acceptor, state in self.acceptor.get_edges(self.target_state):
                logger.debug(
                    "\n%sDownstream edge analysis:\n"
                    f"  Source state: {self.target_state}\n"
                    f"  Target state: {state}\n"
                    f"  Acceptor: {acceptor.__class__.__name__}\n"
                    f"  Acceptor details: {pformat(acceptor, indent=2)}",
                    depth_prefix,
                )
                for possible_next_walker in acceptor.get_walkers():
                    prefixes = possible_next_walker.get_valid_continuations(
                        dawg, depth + 1
                    )
                    edge_walker_prefixes.update(prefixes)

        logger.debug(
            "\nSelecting prefixes at depth %d:\n"
            f"  Current state: {self.current_state}\n"
            f"  Target state: {self.target_state}\n"
            f"  Valid prefixes: {pformat(valid_prefixes, indent=2)}\n"
            f"  Edge walker prefixes: {pformat(edge_walker_prefixes, indent=2)}"
        )
        if not edge_walker_prefixes:
            return valid_prefixes
        return valid_prefixes.union(edge_walker_prefixes)
