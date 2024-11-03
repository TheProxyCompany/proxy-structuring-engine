from __future__ import annotations
from typing import Iterable, Set

from lexpy import DAWG
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.walker import Walker
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

    def get_walkers(self) -> Iterable[Walker]:
        """Initialize walkers at the start state.

        Returns:
            Iterable of walkers positioned at initial state.
        """
        yield self._walker(self)

    def expects_more_input(self, walker: Walker) -> bool:
        return (
            walker._accepts_remaining_input
            and walker.consumed_character_count < self.max_whitespace
        )

    def is_optional(self) -> bool:
        return self.min_whitespace == 0


class WhitespaceWalker(Walker):
    """
    Walker for WhitespaceAcceptor utilizing TokenTrie.
    """

    def __init__(self, acceptor: WhitespaceAcceptor, text: str = ""):
        """
        Initialize the walker.

        Args:
            acceptor (WhitespaceAcceptor): The parent acceptor.
            text (str, optional): Accumulated whitespace text. Defaults to "".
        """
        super().__init__(acceptor)
        self.acceptor: WhitespaceAcceptor = acceptor
        self.text: str = text
        self.length_exceeded: bool = len(text) > self.acceptor.max_whitespace
        self._accepts_remaining_input: bool = True

    def should_start_transition(self, token: str) -> bool:
        if not token:
            return False

        return token.isspace() or token[0].isspace()

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
            self._accepts_remaining_input
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
        logger.debug(f"input: {repr(token)}, advancing walker: {self}")
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

        if remaining_input:
            logger.debug(f"remaining_input: {repr(remaining_input)}")

        if not valid_input:
            self._accepts_remaining_input = False
            logger.debug("no valid whitespace prefix, returning no walkers")
            if remaining_input and len(self.text) >= self.acceptor.min_whitespace:
                copy = WhitespaceWalker(self.acceptor, self.text)
                copy.remaining_input = remaining_input
                copy._accepts_remaining_input = False
                yield AcceptedState(copy)
            return

        logger.debug(f"valid_input: {repr(valid_input)}")

        # Check if the length exceeds the maximum allowed whitespace
        if len(self.text + valid_input) > self.acceptor.max_whitespace:
            self.length_exceeded = True
            self._accepts_remaining_input = False
            return []

        next_walker = WhitespaceWalker(self.acceptor, self.text + valid_input)
        # whitespace acceptor shouldn't accept non-whitespace remaining input
        # if any is found, return it but set self to
        # not accept remaining input
        next_walker._accepts_remaining_input = remaining_input is None
        next_walker.consumed_character_count += valid_length

        if (
            len(next_walker.text) >= self.acceptor.min_whitespace
            and remaining_input is None
        ):
            yield AcceptedState(next_walker)
        elif remaining_input:
            next_walker.remaining_input = remaining_input
            yield next_walker

    def accumulated_value(self) -> str:
        """
        Get the accumulated whitespace value.

        Returns:
            str: The whitespace string.
        """
        return self.text

    def is_within_value(self) -> bool:
        return len(self.text) > 0
