from __future__ import annotations

import logging
from pprint import pformat
from typing import Iterable, Optional, Set, Tuple, Type

from lexpy import DAWG

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.types import (
    EdgeType,
    MAX_LOOKAHEAD_DEPTH,
    StateMachineGraph,
    StateType,
)
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
            graph: A dictionary mapping states to lists of (TokenAcceptor, target_state) tuples.
                   Defaults to empty dict if not provided.
            initial_state: The starting state. Defaults to 0.
            end_states: Collection of accepting states. Defaults to ["$"].
            is_optional: Whether the acceptor is optional. Defaults to False.
            is_case_sensitive: Whether the acceptor is case sensitive. Defaults to True.
        """
        super().__init__(
            initial_state,
            end_states or ["$"],
            walker_type or StateMachineWalker,
            is_optional,
            is_case_sensitive,
        )
        self.graph: StateMachineGraph = graph or {}

    def _branch_walkers(
        self, walker: Walker, token: Optional[str] = None
    ) -> Iterable[Walker]:
        """Branch the walker into multiple paths for parallel exploration.

        At each non-deterministic decision point, clone the current walker
        for each possible transition. This allows simultaneous exploration
        of all valid paths without explicit backtracking.

        Args:
            walker: The current walker instance to branch from.

        Yields:
            New walker instances, each representing a different path.
        """
        for transition, target_state in self._get_transitions(
            walker.current_state, walker
        ):
            if token and not transition.should_start_transition(token):
                logger.debug(
                    f"Transition {transition} cannot start with {token}, skipping."
                )
                continue

            if walker.remaining_input and not transition.should_start_transition(
                walker.remaining_input
            ):
                logger.debug(
                    "Walker has remaining input but transition cannot start, dead end."
                )
                continue

            if (
                walker.target_state == target_state
                and walker.transition_walker
                and walker.transition_walker.can_accept_more_input()
            ):
                logger.debug(f"Walker already exploring state {target_state}")
                yield walker
                continue

            if (
                walker.transition_walker
                and walker.transition_walker.has_reached_accept_state()
            ):
                walker.accepted_history.append(walker.transition_walker)
                logger.debug(
                    f"Appended transition walker to accept history: {walker.accepted_history}"
                )

            if (
                transition.has_reached_accept_state()
                and target_state in self.end_states
            ):
                logger.debug(
                    f"Transition has reached accept state, yielding accepted state: {target_state}"
                )
                yield AcceptedState(walker)
                continue

            new_walker = walker.clone()
            new_walker.transition_walker = transition
            new_walker.target_state = target_state

            yield new_walker

    def _get_transitions(
        self,
        state: StateType,
        source_walker: Optional[Walker] = None,
    ) -> Iterable[Tuple[Walker, StateType]]:
        """Get transition walkers for a given state.

        Returns:
            Iterable of (transition_walker, target_state) tuples.
        """
        for acceptor, target_state in self.get_edges(state):
            logger.debug(f"Exploring edge ({state})--[{acceptor}]-->({target_state})")
            for walker in acceptor.get_walkers():
                yield walker, target_state

            if (
                acceptor.is_optional()
                # or (source_walker and source_walker.has_reached_accept_state())
            ):
                if (
                    target_state in self.end_states
                    and source_walker
                    and not source_walker.can_accept_more_input()
                ):
                    logger.debug(
                        f"{target_state} is in {self.end_states}, yielding accepted walker: {source_walker}"
                    )
                    yield AcceptedState(source_walker), target_state
                else:
                    logger.debug(
                        f"State {state} supports pass-through, getting transitions for next state {target_state}"
                    )
                    yield from self._get_transitions(target_state, source_walker)

    def _transition(self, walker: Walker) -> Iterable[Walker]:
        """Handle transitions reaching accepted states.

        Manages the transition cascade process:
        1. Complete current transition
        2. Update walker state and history
        3. Find new transitions if not in end state

        Args:
            walker: Walker that reached accepted state

        Yields:
            Walkers after transition

        Raises:
            ValueError: If walker lacks required transition state
        """
        logger.debug(f"Transitioning this walker:\n{walker}")
        if not walker.transition_walker or walker.target_state is None:
            logger.error(
                f"Walker lacks transition_walker or target_state:\n{pformat(walker, indent=2)}\n"
            )
            raise ValueError("Walker must have a transition_walker and target_state.")

        transition_value = walker.transition_walker.raw_value
        is_end_state = walker.target_state in self.end_states

        if not walker.should_complete_transition(transition_value, is_end_state):
            logger.debug(
                f"Could not complete transition with {transition_value} for walker:\n{walker}"
            )
            yield walker
            return

        yield walker.transition()

    @classmethod
    def advance_all(
        cls,
        walkers: Iterable[Walker],
        token: str,
    ) -> Iterable[Walker]:
        logger.warning("advance_all is deprecated, use advance_all_walkers instead")
        for _, walker in cls.advance_all_walkers(walkers, token):
            yield walker

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

        def process_walker(walker: Walker) -> Iterable[Walker]:
            """Process a single walker by advancing it with the given input.

            Args:
                walker: The walker to process.

            Returns:
                Updated walkers after advancement.
            """
            yield from walker.consume_token(token)

        # Using map with executor and yielding results
        for advanced_walkers in cls._EXECUTOR.map(process_walker, walkers):
            for walker in advanced_walkers:
                if not walker.remaining_input:
                    logger.debug(f"Advanced input: {repr(token)}")
                    yield token, walker
                    continue

                # Partial match
                length_of_remaining_input = len(token) - len(walker.remaining_input)
                valid_prefix = token[:length_of_remaining_input]
                logger.debug(f"potential partial match: {repr(valid_prefix)}")
                if dawg and valid_prefix and valid_prefix in dawg:
                    logger.debug(f"Valid partial match: {repr(valid_prefix)}, token found in dawg")
                    walker.remaining_input = None
                    yield valid_prefix, walker

    def advance_walker(self, walker: Walker, token: str) -> Iterable[Walker]:
        """Advance walker state with input token.

        Processes the input token and manages state transitions:
        1. If no active transition, yield current walker if more input expected.
        2. Otherwise, advance transition walker and cascade resulting states.
        3. Handle backtracking if advancement fails.

        Args:
            walker: Walker to advance.
            token: Input string to process.

        Yields:
            Updated walkers after processing token.
        """
        if not walker.should_start_transition(token):
            logger.debug(f"{walker} cannot start transition with {token}")
            if walker.remaining_input:
                logger.debug("Yielding walker upstream for partial matching")
                yield walker

            if walker.transition_walker and walker.transition_walker.has_reached_accept_state():
                logger.debug("Walker has reached accept state, branching to next states")
                # did_branch_walker = False
                for next_walker in self._branch_walkers(walker):
                    # did_branch_walker = True
                    if next_walker.should_start_transition(token):
                        yield from self.advance_walker(next_walker, token)
            return

        if not walker.transition_walker:
            logger.debug(f"Branching walkers with {repr(token)} from {walker}")

            did_branch_walker = False
            for next_walker in self._branch_walkers(walker):
                did_branch_walker = True
                yield from self.advance_walker(next_walker, token)

            if not did_branch_walker and walker.remaining_input:
                logger.debug(f"Walker could not find valid branches:\n{walker}")
                logger.debug("Yielding walker upstream for partial matching")
                yield walker

            return

        if (
            walker.transition_walker
            and walker.accepted_history
            and walker.transition_walker == walker.accepted_history[-1]
        ):
            logger.debug(
                f"Popping accepted walker from history: {walker.accepted_history[-1]}"
            )
            walker.accepted_history.pop()

        for advanced_walker in walker.transition_walker.consume_token(token):
            logger.debug(f"Advanced transition walker: {advanced_walker}")

            new_walker = walker.clone()
            new_walker.remaining_input = advanced_walker.remaining_input
            new_walker.transition_walker = advanced_walker
            new_walker.transition_walker.remaining_input = None
            new_walker.consumed_character_count += (
                advanced_walker.consumed_character_count
            )

            for next_walker in self._transition(new_walker):
                if not next_walker.remaining_input:
                    logger.debug(
                        f"{walker.acceptor.__class__.__name__}.Walker completely consumed {repr(token)}"
                    )
                    yield next_walker
                    continue

                logger.debug(
                    f"{walker.__class__.__name__}.Walker has remaining input, recursively advancing"
                )
                yield from self.advance_walker(next_walker, next_walker.remaining_input)

    def get_edges(self, state: StateType) -> Iterable[EdgeType]:
        """Retrieve outgoing transitions for a given state.

        Args:
            state: Source state to get transitions from.

        Returns:
            List of (acceptor, target_state) tuples representing possible transitions.
        """
        return self.graph.get(state, [])

    def get_walkers(self, state: Optional[StateType] = None) -> Iterable[Walker]:
        """Initialize walkers at the start state.

        Args:
            state: Optional starting state.

        Returns:
            Iterable of walkers positioned at the initial state.
        """
        initial_walker = self._walker(self, state)
        yield from self._branch_walkers(initial_walker)


class StateMachineWalker(Walker):
    """Walker for navigating through StateMachine states.

    Manages state traversal and tracks:
    - Current position
    - Transition state
    - Input processing
    - Value accumulation
    """

    def accepts_any_token(self) -> bool:
        """Check if current transition matches all characters.

        Returns:
            True if matches all, False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.accepts_any_token()
        return False

    def can_accept_more_input(self) -> bool:
        """Check if walker can process more input.

        Returns:
            True if more input can be handled, False otherwise.
        """
        if self.transition_walker:
            return self.transition_walker.can_accept_more_input()

        if not self._accepts_remaining_input:
            return False

        if isinstance(self.acceptor, StateMachine):
            return bool(self.acceptor.graph.get(self.current_state))

        return False

    def consume_token(self, token: str) -> Iterable[Walker]:
        """Advance walker with input token.

        Args:
            token: Input to process.

        Yields:
            Updated walkers after advancement.
        """
        yield from self.acceptor.advance_walker(self, token)

    def is_within_value(self) -> bool:
        """Determine if the walker is currently within a value.

        Returns:
            True if in a value, False otherwise.
        """
        if self.consumed_character_count > 0 and self.transition_walker:
            return self.transition_walker.is_within_value()

        return False

    def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        """Select valid characters based on current state.

        Args:
            dawg: DAWG to select from.
            depth: Current depth in state machine.

        Returns:
            Set of valid characters.
        """
        if depth >= MAX_LOOKAHEAD_DEPTH:
            return set()

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
