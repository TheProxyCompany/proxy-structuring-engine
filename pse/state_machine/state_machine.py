from __future__ import annotations

import logging
from pprint import pformat
from typing import Iterable, Optional, Set, Tuple, Type
from collections import deque

from lexpy import DAWG

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.types import (
    EdgeType,
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
                Defaults to an empty dict if not provided.
            initial_state: The starting state. Defaults to 0.
            end_states: Collection of accepting states. Defaults to ["$"].
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
            state: Source state to get transitions from.

        Returns:
            Iterable of (acceptor, target_state) tuples representing possible transitions.
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
        yield from self.branch_walkers(initial_walker)

    def get_transitions(
        self,
        state: StateType,
        source_walker: Optional[Walker] = None,
    ) -> Iterable[Tuple[Walker, StateType]]:
        """Get transition walkers for a given state.

        Args:
            state: The current state.
            source_walker: The walker initiating the transition.

        Returns:
            Iterable of (transition_walker, target_state) tuples.
        """
        for acceptor, target_state in self.get_edges(state):
            for walker in acceptor.get_walkers():
                yield walker, target_state

            if acceptor.is_optional():
                if (
                    target_state in self.end_states
                    and source_walker
                    and not source_walker.can_accept_more_input()
                ):
                    logger.debug(
                        f"🟢 {target_state} is in {self.end_states}, yielding accepted walker: {source_walker}"
                    )
                    yield AcceptedState(source_walker), target_state
                else:
                    logger.debug(
                        f"🟡 edge ({state})--[{acceptor}]-->({target_state}) supports pass-through, getting transitions for next state {target_state}"
                    )
                    yield from self.get_transitions(target_state, source_walker)

    def branch_walkers(
        self,
        walker: Walker,
        token: Optional[str] = None,
    ) -> Iterable[Walker]:
        """Branch the walker into multiple paths for parallel exploration.

        At each non-deterministic decision point, clone the current walker
        for each possible transition. This allows simultaneous exploration
        of all valid paths without explicit backtracking.

        Args:
            walker: The current walker instance to branch from.
            token: Optional token to start transitions.

        Yields:
            New walker instances, each representing a different path.
        """
        for transition, target_state in self.get_transitions(
            walker.current_state, walker
        ):
            # Skip if transition can't start with either token or walker's remaining input
            # saves time by skipping transitions that can't be completed
            if (token and not transition.should_start_transition(token)) or (
                walker.remaining_input
                and not transition.should_start_transition(walker.remaining_input)
            ):
                logger.debug(
                    "🔴 Skipping %s - cannot start with %s",
                    transition.acceptor,
                    repr(token or walker.remaining_input),
                )
                continue

            if (
                walker.target_state == target_state
                and walker.transition_walker
                and walker.transition_walker.can_accept_more_input()
            ):
                logger.debug(
                    f"🟡 {walker.acceptor}.Walker already exploring state {target_state}, skipping {transition.acceptor}"
                )
                continue

            yield walker.set_target(transition, target_state)

    def advance_walker(self, walker: Walker, token: str) -> Iterable[Walker]:
        """Advance the walker state with the given input token.

        Processes the input token and manages state transitions:
        1. If no active transition, branches to next possible states.
        2. If an active transition walker can proceed, advances it.
        3. If no valid transitions and the walker can accept more input, yields the walker.

        Args:
            walker: The walker to advance.
            token: The input string to process.

        Yields:
            Updated walkers after processing the token.
        """
        queue: deque[Tuple[Walker, str]] = deque([(walker, token)])

        while queue:
            current_walker, current_token = queue.popleft()

            if current_walker.transition_walker is None:
                logger.debug(
                    "🔵 Walker has no transition walker. Branching to next states."
                )
                # breakpoint()
                has_valid_transition = False
                for next_walker in self.branch_walkers(current_walker, current_token):
                    has_valid_transition = True
                    queue.append((next_walker, current_token))

                if not has_valid_transition and current_walker.remaining_input:
                    logger.debug(
                        f"🟢 Yielding walker with remaining input: {current_walker}"
                    )
                    yield current_walker

                continue

            if current_walker.transition_walker.should_start_transition(current_token):
                if (
                    current_walker.accepted_history
                    and current_walker.transition_walker
                    == current_walker.accepted_history[-1]
                ):
                    logger.debug(
                        f"🟡 Popping accepted walker from history: {current_walker.accepted_history[-1]}"
                    )
                    current_walker.accepted_history.pop()

                for advanced_walker in current_walker.transition_walker.consume_token(
                    current_token
                ):
                    new_walker = current_walker.transition_with_walker(advanced_walker)

                    if new_walker.remaining_input:
                        logger.debug(f"⚪️ walker with remaining input: {new_walker}")
                        queue.append((new_walker, new_walker.remaining_input))
                    else:
                        logger.debug(f"🟢 walker: {new_walker}")
                        yield new_walker
                continue

            logger.debug(
                f"🔴 {current_walker.transition_walker.acceptor} cannot start transition with {repr(current_token)}"
            )

            if current_walker.transition_walker.can_accept_more_input():
                has_valid_transition = False
                for next_transition_walker in current_walker.transition_walker.branch():
                    if next_transition_walker.should_start_transition(current_token):
                        has_valid_transition = True
                        current_walker.transition_walker = next_transition_walker
                        queue.append((current_walker, current_token))

                if has_valid_transition:
                    continue

            if current_walker.transition_walker.has_reached_accept_state():
                logger.debug(
                    f"🟢 {current_walker.transition_walker.acceptor} has reached accept state"
                )
                logger.debug(f"🔵 Branching {current_walker.acceptor} to next states.")
                for next_walker in self.branch_walkers(current_walker):
                    if next_walker.should_start_transition(current_token):
                        queue.append((next_walker, current_token))


        # after queue is empty
        logger.debug(f"🔵 {self} queue is empty!")

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
            logger.debug(f"⚪️ Starting to process walker with token: {repr(token)}")
            yield from walker.consume_token(token)

        # Using map with executor and yielding results
        for advanced_walkers in cls._EXECUTOR.map(process_walker, walkers):
            for walker in advanced_walkers:
                if not walker.remaining_input:
                    logger.debug(f"🟢 Advanced input: {repr(token)}")
                    yield token, walker
                    continue

                # Partial match
                length_of_remaining_input = len(token) - len(walker.remaining_input)
                valid_prefix = token[:length_of_remaining_input]

                logger.debug(f"🟡 Potential partial match: {repr(valid_prefix)}")

                if dawg is not None and valid_prefix and valid_prefix in dawg:
                    logger.debug(
                        f"🟢 Valid partial match: {repr(valid_prefix)}, token found in DAWG"
                    )
                    walker.remaining_input = None
                    yield valid_prefix, walker

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
            if isinstance(acceptor, StateMachine):
                acceptor_repr = acceptor.__repr__()
                return "\n".join(
                    ("    " * indent + line) if idx != 0 else line
                    for idx, line in enumerate(acceptor_repr.splitlines())
                )
            return repr(acceptor)

        formatted_graph = format_graph(self.graph)
        return f"{self.__class__.__name__}({formatted_graph})"


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
                logger.debug(
                    f"🟢 {self.transition_walker.acceptor} can accept more input"
                )
                return True

            logger.debug(
                f"🔴 {self.transition_walker.acceptor} cannot accept more input"
            )

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
