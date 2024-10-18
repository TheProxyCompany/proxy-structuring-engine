"""
Base Token Acceptors Module.

This module defines the foundational classes and methods for token acceptors,
which constrain the tokens acceptable during parsing or generation of text.
Token acceptors utilize cursors to manage multiple parsing states efficiently,
minimizing expensive backtracking operations.

Classes:
    TokenAcceptor: Base class for all token acceptors.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterable, Type

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pse.state_machine.cursor import Cursor
    from pse.state_machine.types import StateType


class TokenAcceptor(ABC):
    """Base class for token acceptors.

    A token acceptor constrains the acceptable tokens at a specific point
    during parsing or generation. It manages multiple cursors representing
    different valid states, enabling efficient traversal and minimizing
    backtracking.

    Attributes:
        initial_state (StateType): The starting state of the acceptor.
        end_states (Iterable[StateType]): A collection of acceptable end states.
    """

    _MAX_WORKERS: int = min(32, os.cpu_count() or 1)
    _EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

    def __init__(
        self,
        initial_state: StateType,
        end_states: Iterable[StateType],
    ) -> None:
        """Initializes the TokenAcceptor with the given initial and end states.

        Args:
            initial_state (StateType): The starting state of the acceptor.
            end_states (Iterable[StateType]): A collection of acceptable end states.
        """
        self.initial_state = initial_state
        self.end_states = end_states

    @property
    @abstractmethod
    def cursor_class(self) -> Type[Cursor]:
        """Retrieves the cursor class associated with this acceptor.

        Returns:
            Type[Cursor]: The cursor class.
        """
        pass

    @abstractmethod
    def advance_cursor(self, cursor: Cursor, input_str: str) -> Iterable[Cursor]:
        """Advances the cursor with the given input.

        Args:
            cursor (Cursor): The cursor to advance.
            input_str (str): The input string to process.

        Returns:
            Iterable[Cursor]: An iterable of updated cursors after advancement.
        """
        pass

    @abstractmethod
    def expects_more_input(self, cursor: Cursor) -> bool:
        """Checks if the acceptor expects more input after the current cursor position.

        Args:
            cursor (Cursor): The cursor to check.

        Returns:
            bool: True if more input is expected, False otherwise.
        """
        pass

    @classmethod
    def advance_all(cls, cursors: Iterable[Cursor], input_str: str) -> Iterable[Cursor]:
        """Advances multiple cursors in parallel based on the input.

        Args:
            cursors (Iterable[Cursor]): An iterable of cursor instances.
            input_str (str): The input string to advance the cursors with.

        Returns:
            Iterable[Cursor]: An iterable of new cursor instances after advancement.
        """
        if not cursors:
            return []

        def process_cursor(cursor: Cursor) -> Iterable[Cursor]:
            """Processes a single cursor by advancing it with the given input.

            Args:
                cursor (Cursor): The cursor to process.

            Returns:
                Iterable[Cursor]: Updated cursors after advancement.
            """
            yield from cursor.advance(input_str)

        # Using map with executor and yielding results
        for result in cls._EXECUTOR.map(process_cursor, cursors):
            yield from result

    def get_cursors(self) -> Iterable[Cursor]:
        """Retrieves cursors to traverse the acceptor.

        Returns:
            Iterable[Cursor]: An iterable of cursor instances.
        """
        yield self.cursor_class(self)

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the instance.

        Returns:
            str: The string representation.
        """
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Returns a readable string representation of the instance.

        Returns:
            str: The string representation.
        """
        return repr(self)
