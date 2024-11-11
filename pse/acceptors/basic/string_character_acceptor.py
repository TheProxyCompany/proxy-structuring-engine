from __future__ import annotations
from typing import Iterable, Optional, Set

from lexpy import DAWG

from pse.state_machine.state_machine import StateMachine
from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.walker import Walker, logger

# INVALID_CHARS is a set containing characters that are not allowed in JSON strings.
# It includes control characters (ASCII 0-31) and the double quote (") and backslash (\) characters.
# These characters need to be escaped or are not permitted in JSON string values.
INVALID_CHARS: Set[str] = {chr(c) for c in range(0, 0x20)} | {'"', "\\"}


class StringCharacterAcceptor(StateMachine):
    """
    Accepts one or more valid JSON unescaped string characters.
    """

    def __init__(self) -> None:
        """
        Initialize the StringCharAcceptor with its state transitions.
        """
        super().__init__(
            initial_state=0, end_states={1}, walker_type=StringCharacterWalker
        )

    def get_walkers(self) -> Iterable[Walker]:
        yield self._walker(self)

    @classmethod
    def prepare_dawg(cls, dawg: DAWG) -> DAWG:
        """
        Build a collapsed trie that reduces the search space for valid tokens.
        Multiple collapsed tries are cached to handle scenarios where string
        matching starts in the middle of the main trie.

        Args:
            dawg (DAWG): The main vocabulary dawg.

        Returns:
            DAWG: The optimized string character dawg.
        """
        cls.valid_chars = set(INVALID_CHARS)
        return dawg


class StringCharacterWalker(Walker):
    """
    Walker for navigating through characters in StringCharAcceptor.
    """

    def __init__(
        self, acceptor: StringCharacterAcceptor, value: Optional[str] = None
    ) -> None:
        """
        Initialize the Walker.

        Args:
            acceptor (StringCharAcceptor): The parent StringCharAcceptor.
            value (Optional[str]): The accumulated string value. Defaults to None.
        """
        super().__init__(acceptor)
        self.acceptor: StringCharacterAcceptor = acceptor
        self._accepts_remaining_input = True
        self._raw_value = value

    def can_accept_more_input(self) -> bool:
        return self._accepts_remaining_input

    def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        """
        Select valid string characters by excluding invalid ones.

        Returns:
            Set[str]: Set of valid string characters.
        """
        return self.acceptor.valid_chars

    def should_start_transition(self, token: str) -> bool:
        if not token:
            return False
        return token[0] not in INVALID_CHARS

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advance the walker with the given input.

        Args:
            input (str): The input to advance with.

        Returns:
            List[Walker]: List of new walkers after advancement.
        """
        logger.debug(
            f"Advancing walker in string char acceptor: {self}, with input: {token}"
        )
        # clean the input of invalid characters
        valid_prefix = ""
        remaining_input = token

        for index, char in enumerate(token):
            if char not in INVALID_CHARS:
                valid_prefix += char
            else:
                remaining_input = token[index:]
                break
        else:
            remaining_input = None

        if valid_prefix:
            new_walker = self.clone()
            new_walker._raw_value = (self._raw_value or "") + valid_prefix
            new_walker.remaining_input = remaining_input
            new_walker.consumed_character_count += len(valid_prefix)
            yield AcceptedState(new_walker)
        else:
            self._accepts_remaining_input = False
            logger.debug(f"{self} cannot accept more input: {token}")

    def is_within_value(self) -> bool:
        """
        Check if the walker has a value.

        Returns:
            bool: True if the walker has a value, False otherwise.
        """
        return self._raw_value is not None
