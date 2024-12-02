from __future__ import annotations
from typing import Any, Optional, List, Set, Tuple, Iterable, Dict, Type
from abc import abstractmethod

State = int | str
"""Represents a state within the StateMachine.  Can be an integer or a string."""

Edge = Tuple[Acceptor, State]
"""Represents a transition between states in the StateMachine.

An Edge consists of a `TokenAcceptor` that governs the conditions for the transition,
and the `StateType` of the destination state.  Walkers traverse edges based on
the acceptance criteria of the associated `TokenAcceptor`.
"""

VisitedEdge = Tuple[State, Optional[State], Optional[str]]
"""Represents a traversed edge during StateMachine execution.

A `VisitedEdgeType` records the source state, target state, and the token
consumed during the transition.  This history aids in tracking walker progress
and preventing cycles.
"""

StateGraph = Dict[State, List[Edge]]
"""Represents the structure of the StateMachine as a directed graph.

Maps each `State` to a list of its outgoing `Edge`s.  This graph
defines the possible transitions between states and forms the basis for
walker navigation.
"""

class Walker:
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
    acceptor: Acceptor
    current_state: State
    target_state: Optional[State]
    transition_walker: Optional[Walker]
    consumed_character_count: int
    remaining_input: Optional[str]
    explored_edges: Set[VisitedEdge]
    accepted_history: List[Walker]
    _accepts_more_input: bool
    _raw_value: Optional[str]

    def __init__(
        self,
        acceptor: Acceptor,
        current_state: Optional[State] = None
    ) -> None: ...

    @abstractmethod
    def consume_token(self, token: str) -> Iterable[Walker]: ...

    @abstractmethod
    def can_accept_more_input(self) -> bool: ...

    @abstractmethod
    def is_within_value(self) -> bool: ...

    def clone(self) -> Walker: ...
    def should_start_transition(self, token: str) -> bool: ...
    def should_complete_transition(self) -> bool: ...

    def start_transition(
        self,
        transition_walker: Walker,
        token: Optional[str] = None,
        start_state: Optional[State] = None,
        target_state: Optional[State] = None
    ) -> Optional[Walker]: ...

    def complete_transition(
        self,
        transition_walker: Walker
    ) -> Tuple[Optional[Walker], bool]: ...

    def branch(self, token: Optional[str] = None) -> Iterable[Walker]: ...
    def accepts_any_token(self) -> bool: ...
    def get_valid_continuations(self, depth: int = 0) -> Iterable[str]: ...
    def find_valid_prefixes(self, dawg: Any) -> Set[str]: ...
    def has_reached_accept_state(self) -> bool: ...

    @property
    def current_value(self) -> Any: ...
    @property
    def raw_value(self) -> Optional[str]: ...
    @property
    def current_edge(self) -> VisitedEdge: ...

    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class Acceptor:
    """
    Base class for token acceptors.

    An acceptor constrains the acceptable tokens at a specific point
    during parsing or generation.

    It manages multiple walkers representing different valid states,
    enabling efficient traversal and minimizing backtracking.

    Attributes:
        initial_state (StateType): The starting state of the acceptor.
        end_states (Iterable[StateType]): A collection of acceptable end states.
    """

    state_graph: StateGraph
    initial_state: State
    end_states: Set[State]

    def __init__(
        self,
        state_graph: Optional[StateGraph] = None,
        start_state: State = 0,
        end_states: Optional[Iterable[State]] = None,
        is_optional: bool = False,
        is_case_sensitive: bool = True,
    ) -> None:
        """Initialize the Acceptor with a state graph.

        Args:
            state_graph: Mapping of states to lists of (TokenAcceptor, target_state) tuples.
                  Defaults to an empty dictionary.
            start_state: The starting state. Defaults to 0.
            end_states: Collection of accepting states. Defaults to ("$",).
            is_optional: Whether the acceptor is optional. Defaults to False.
            is_case_sensitive: Whether the acceptor is case sensitive. Defaults to True.
        """
        ...

    @property
    def is_optional(self) -> bool:
        """Checks if the acceptor is optional.

        Returns:
            bool: True if the acceptor is optional, False otherwise.
        """
        ...

    @property
    def is_case_sensitive(self) -> bool:
        """Checks if the acceptor is case sensitive.

        Returns:
            bool: True if the acceptor is case sensitive, False otherwise.
        """
        ...

    @property
    @abstractmethod
    def walker_class(self) -> Type[Walker]:
        """The walker class for this acceptor."""
        ...

    @abstractmethod
    def get_walkers(
        self,
        state: Optional[State] = None,
    ) -> Iterable[Walker]:
        """Retrieves walkers to traverse the acceptor.

        Returns:
            Iterable[Walker]: An iterable of walker instances.
        """
        ...

    @abstractmethod
    def get_transitions(
        self,
        walker: Walker,
    ) -> Iterable[Tuple[Walker, State]]:
        """Retrieves transitions from the given walker."""
        ...

    @abstractmethod
    def advance(
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
        ...

    @abstractmethod
    def branch_walker(
        self,
        walker: Walker,
        token: Optional[str] = None,
    ) -> Iterable[Walker]:
        """Branch the walker into multiple paths for parallel exploration."""
        ...

    def __repr__(self) -> str:
        """Return a formatted string representation of the StateMachine instance.

        This method provides a detailed view of the state machine's configuration,
        formatted with proper indentation for better readability.

        Returns:
            str: A formatted string showing the state machine's configuration.
        """
        ...
