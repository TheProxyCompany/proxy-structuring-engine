from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import copy as shallow_copy
from typing import Any, Iterable, List, Optional, Self, Set, TYPE_CHECKING
from pprint import pformat

from lexpy import DAWG

from pse.state_machine.types import StateType

if TYPE_CHECKING:
    from pse.acceptors.token_acceptor import TokenAcceptor

logger = logging.getLogger(__name__)

MAX_DEPTH = 3


class Cursor(ABC):
    """Represents the current state in a token acceptor's state machine.

    Cursors manage progression through different states as input is consumed during parsing or generation.

    Attributes:
        acceptor: The token acceptor instance associated with this cursor.
        current_state: The current state in the state machine.
        target_state: The target state for transitions, if any.
        transition_cursor: The cursor handling the current transition.
        accept_history: A list of cursors representing the history of accepted states.
        consumed_character_count: The number of characters consumed so far.
        remaining_input: The remaining input string to be processed.
        _accepts_remaining_input: Indicates if the cursor can accept more input.
    """

    def __init__(self, acceptor: TokenAcceptor) -> None:
        """Initializes a new Cursor with the given acceptor.

        Args:
            acceptor: The token acceptor instance.
        """
        self.acceptor = acceptor
        self.current_state: StateType = acceptor.initial_state
        self.target_state: Optional[StateType] = None
        self.transition_cursor: Optional[Cursor] = None
        self.accept_history: List[Cursor] = []
        self.consumed_character_count: int = 0
        self.remaining_input: Optional[str] = None
        self._accepts_remaining_input: bool = False

    # -------- Abstract Methods --------

    @abstractmethod
    def get_value(self) -> Any:
        """Retrieves the current value accumulated by the cursor.

        Returns:
            The current accumulated value.
        """
        pass

    @abstractmethod
    def advance(self, input_str: str) -> Iterable[Cursor]:
        """Advances the cursor with the given input string.

        Args:
            input_str: The input string to process.

        Yields:
            Updated cursor instances after advancement.
        """
        pass

    @abstractmethod
    def is_in_value(self) -> bool:
        """Determines if the cursor is currently within a value.

        Returns:
            True if in a value; False otherwise.
        """
        pass

    # -------- Public Methods --------

    @property
    def can_handle_remaining_input(self) -> bool:
        """Indicates whether the cursor can handle additional input.

        Returns:
            True if the cursor can handle more input; False otherwise.
        """
        return self._accepts_remaining_input

    def start_transition(
        self,
        transition_acceptor: TokenAcceptor,
        target_state: StateType,
    ) -> bool:
        """Determines if a transition should start with the given acceptor and target state.

        Args:
            transition_acceptor: The acceptor initiating the transition.
            target_state: The target state for the transition.

        Returns:
            True if the transition should start; False otherwise.
        """
        return True

    def complete_transition(
        self,
        transition_value: Any,
        target_state: StateType,
        is_end_state: bool,
    ) -> bool:
        """Determines if the transition should complete with the given parameters.

        Args:
            transition_value: The value accumulated during the transition.
            target_state: The target state after the transition.
            is_end_state: Indicates if the target state is an end state.

        Returns:
            True if the transition should complete; False otherwise.
        """
        return True

    def clone(self) -> Self:
        """Creates a shallow copy of the cursor, including its accept history.

        Returns:
            A new Cursor instance that is a clone of the current one.
        """
        cloned_cursor = shallow_copy(self)
        cloned_cursor.accept_history = self.accept_history.copy()
        return cloned_cursor

    def matches_all(self) -> bool:
        """Checks if the acceptor accepts all tokens (i.e., free text).

        Returns:
            True if all tokens are accepted; False otherwise.
        """
        return False

    def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        """Selects substrings (edges) for cursor advancement based on the DAWG.

        Args:
            dawg: The Directed Acyclic Word Graph to select from.
            depth: The current depth in the state machine traversal.

        Returns:
            A set of selected substrings.
        """
        logger.debug(
            "Selecting substrings for cursor:\n"
            f"  Current State: {self.current_state}\n"
            f"  Depth: {depth}\n"
            f"  DAWG Size: {len(dawg)}"
        )
        return set()

    def get_valid_prefixes(self, dawg: DAWG) -> Set[str]:
        """Identifies tokens that can advance the acceptor to a valid state.

        Args:
            dawg: The Directed Acyclic Word Graph to search for valid prefixes.

        Returns:
            A set of valid prefixes that can be used to advance the cursor.
        """
        valid_prefixes: Set[str] = set()
        selected_substrings = self.select(dawg)

        logger.debug(
            "Processing valid prefixes:\n"
            f"  Cursor: {self.__class__.__name__}\n"
            f"  Selected Substrings:\n{pformat(selected_substrings, indent=4)}"
        )

        for substr in selected_substrings:
            logger.debug(
                f"Finding tokens with prefix: {repr(substr)}\n"
                f"  Current State: {self.current_state}"
            )
            valid_prefixes.update(dawg.search_with_prefix(substr))  # type: ignore

        logger.debug(
            f"Found valid prefixes:\n{pformat(valid_prefixes, indent=4)}"
        )
        return valid_prefixes

    def in_accepted_state(self) -> bool:
        """Checks if the cursor has reached an accepted (final) state.

        Returns:
            True if in an accepted state; False otherwise.
        """
        return False

    # -------- Magic Methods --------

    def __hash__(self) -> int:
        """Generates a hash based on the cursor's state and value.

        Returns:
            An integer hash value.
        """
        return hash((self.current_state, self.target_state, str(self.get_value())))

    def __eq__(self, other: Any) -> bool:
        """Checks equality based on the cursor's state and accumulated value.

        Args:
            other: The object to compare with.

        Returns:
            True if both cursors are equal; False otherwise.
        """
        return (
            isinstance(other, Cursor)
            and self.current_state == other.current_state
            and self.target_state == other.target_state
            and self.get_value() == other.get_value()
        )

    def __repr__(self) -> str:
        """Provides a detailed string representation of the cursor."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    value={pformat(self.get_value(), indent=4)},\n"
            f"    current_state={self.current_state!r},\n"
            f"    target_state={self.target_state!r},\n"
            f"    acceptor={self.acceptor!r}\n"
            ")"
        )

    def __str__(self) -> str:
        """Provides a concise string representation of the cursor."""
        return (
            f"{self.__class__.__name__}("
            f"state={self.current_state}, "
            f"value={self.get_value()!r})"
        )
