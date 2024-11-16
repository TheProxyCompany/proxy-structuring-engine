from __future__ import annotations
from typing import Iterable, Optional, Set

from lexpy import DAWG
from pse.core.state_machine import StateMachine
from pse.util.state_machine.accepted_state import AcceptedState
from pse.core.walker import Walker
import logging

logger = logging.getLogger()

WHITESPACE_CHARS: str = " \n\r\t"


class WhitespaceAcceptor(StateMachine):
    """
    Optional whitespace acceptor using TokenTrie for efficient matching.
    """

    def __init__(self, min_whitespace: int = 0, max_whitespace: int = 40):
        """
        Initialize the WhitespaceAcceptor.

        Args:
            max_whitespace (int, optional): Maximum allowable whitespace characters.
                                            Defaults to 40.
        """
        super().__init__({}, walker_type=WhitespaceWalker)
        self.min_whitespace: int = min_whitespace
        self.max_whitespace: int = max_whitespace

    @property
    def is_optional(self) -> bool:
        return self.min_whitespace == 0


class WhitespaceWalker(Walker):
    """
    Walker for WhitespaceAcceptor utilizing TokenTrie.
    """

    def __init__(
        self,
        acceptor: WhitespaceAcceptor,
        value: Optional[str] = None,
    ):
        """
        Initialize the walker.

        Args:
            acceptor (WhitespaceAcceptor): The parent acceptor.
            text (str, optional): Accumulated whitespace text. Defaults to "".
        """
        super().__init__(acceptor)
        self.acceptor: WhitespaceAcceptor = acceptor
        self._raw_value = value or ""
        self.consumed_character_count = len(self._raw_value)
        self.length_exceeded: bool = len(self._raw_value) > self.acceptor.max_whitespace
        self._accepts_more_input: bool = (
            len(self._raw_value) <= self.acceptor.max_whitespace
        )

    def should_start_transition(self, token: str) -> bool:
        if not token or not token[0].isspace():
            self._accepts_more_input = False
            return False

        return True

    def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        """
        Select valid whitespace characters.

        Args:
            candidate_chars (Set[str]): Set of candidate characters.

        Returns:
            Set[str]: Set of valid whitespace characters.
        """
        valid_tokens = set()
        if self.length_exceeded:
            return valid_tokens

        for char in set(WHITESPACE_CHARS):
            search_result = dawg.search_with_prefix(char)
            valid_tokens.update(search_result)

        return valid_tokens

    def can_accept_more_input(self) -> bool:
        return (
            self._accepts_more_input
            and self.consumed_character_count < self.acceptor.max_whitespace
        )

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advance the walker with the given characters.
        Args:
            input (str): The characters to advance with.

        Returns:
            List[WhitespaceAcceptor.Walker]: List of new walkers after advancement.
        """
        if self.length_exceeded:
            return []

        # Extract the valid whitespace prefix
        valid_length = 0
        if token.isspace():
            valid_length = len(token)
        else:
            for c in token:
                if c.isspace():
                    valid_length += 1
                else:
                    break

        valid_input = token[:valid_length]
        remaining_input = token[valid_length:] or None

        if not valid_input:
            self._accepts_more_input = False
            if (
                remaining_input
                and self.consumed_character_count >= self.acceptor.min_whitespace
            ):
                copy = self.clone()
                copy.remaining_input = remaining_input
                copy._accepts_more_input = False
                yield AcceptedState(copy)
            return

        next_walker = self.clone()
        next_walker._raw_value = (next_walker._raw_value or "") + valid_input
        next_walker.remaining_input = remaining_input
        next_walker._accepts_more_input = remaining_input is None
        next_walker.consumed_character_count += valid_length

        if next_walker.consumed_character_count > self.acceptor.max_whitespace:
            return []

        if next_walker.consumed_character_count >= self.acceptor.min_whitespace:
            yield AcceptedState(next_walker)
        else:
            yield next_walker

    @property
    def current_value(self) -> str:
        """
        Get the accumulated whitespace value.

        Returns:
            str: The whitespace string.
        """
        return self.raw_value or ""

    def is_within_value(self) -> bool:
        return self.consumed_character_count > 0
