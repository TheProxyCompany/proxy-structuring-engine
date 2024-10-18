from __future__ import annotations
from typing import Iterable, Set

from lexpy import DAWG
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.cursor import Cursor
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
        super().__init__({})
        self.min_whitespace: int = min_whitespace
        self.max_whitespace: int = max_whitespace

    def get_cursors(self) -> Iterable[Cursor]:
        """
        Get one or more cursors to traverse the acceptor.

        Returns:
            Iterable[WhitespaceAcceptor.Cursor]: List of initial cursors.
        """
        cursor = self.Cursor(self)
        if len(cursor.text) >= self.min_whitespace:
            return [AcceptedState(cursor)]
        return [cursor]

    def expects_more_input(self, cursor: Cursor) -> bool:
        return cursor._accepts_remaining_input and cursor.consumed_character_count < self.max_whitespace

    class Cursor(Cursor):
        """
        Cursor for WhitespaceAcceptor utilizing TokenTrie.
        """

        def __init__(self, acceptor: WhitespaceAcceptor, text: str = ""):
            """
            Initialize the cursor.

            Args:
                acceptor (WhitespaceAcceptor): The parent acceptor.
                text (str, optional): Accumulated whitespace text. Defaults to "".
            """
            super().__init__(acceptor)
            self.acceptor: WhitespaceAcceptor = acceptor
            self.text: str = text
            self.length_exceeded: bool = len(text) > self.acceptor.max_whitespace
            self._accepts_remaining_input: bool = True

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

        def advance(self, input: str) -> Iterable[Cursor]:
            """
            Advance the cursor with the given characters.
            Args:
                input (str): The characters to advance with.

            Returns:
                List[WhitespaceAcceptor.Cursor]: List of new cursors after advancement.
            """
            logger.debug(f"Advancing cursor: {self}, input: '{input}'")
            if self.length_exceeded:
                return []

            # Extract the valid whitespace prefix
            valid_length = 0
            if input.isspace():
                valid_length = len(input)
            else:
                for c in input:
                    if c.isspace():
                        valid_length += 1
                    else:
                        break

            valid_prefix = input[:valid_length]
            remaining_input = input[valid_length:] or None

            logger.debug(f"valid_prefix: {repr(valid_prefix)}")
            logger.debug(f"remaining_input: {repr(remaining_input)}")

            if not valid_prefix:
                self._accepts_remaining_input = False
                logger.debug("no valid whitespace prefix, returning no cursors")
                if remaining_input and len(self.text) >= self.acceptor.min_whitespace:
                    copy = WhitespaceAcceptor.Cursor(self.acceptor, self.text)
                    copy.remaining_input = remaining_input
                    copy._accepts_remaining_input = False
                    yield AcceptedState(copy)
                return

            next_text = self.text + valid_prefix

            # Check if the length exceeds the maximum allowed whitespace
            if len(next_text) > self.acceptor.max_whitespace:
                self.length_exceeded = True
                self._accepts_remaining_input = False
                return []

            next_cursor = WhitespaceAcceptor.Cursor(self.acceptor, next_text)
            # whitespace acceptor shouldn't accept non-whitespace remaining input
            # if any is found, return it but set self to
            # not accept remaining input
            next_cursor._accepts_remaining_input = remaining_input is None
            next_cursor.remaining_input = remaining_input
            next_cursor.consumed_character_count += valid_length

            logger.debug(
                f"next_cursor: {next_cursor}, "
                f"remaining_input: {repr(next_cursor.remaining_input)}, "
                f"accepts_remaining_input: {next_cursor._accepts_remaining_input}"
            )

            # Yield AcceptedState if minimum whitespace has been consumed OR no whitespace is required and none was found
            if len(next_text) >= self.acceptor.min_whitespace and next_cursor.remaining_input is None:
                logger.debug(f"yielding accepted state from {self} {AcceptedState(next_cursor)}")
                yield AcceptedState(next_cursor)
            else:
                yield next_cursor

        def get_value(self) -> str:
            """
            Get the accumulated whitespace value.

            Returns:
                str: The whitespace string.
            """
            return self.text

        def is_in_value(self) -> bool:
            return len(self.text) > 0

        def __repr__(self) -> str:
            """
            Return the string representation of the cursor.

            Returns:
                str: The string representation.
            """
            components = []
            if self.get_value():
                components.append(self.get_value())
            if self.transition_cursor:
                components.append(
                    f"state[{self.current_state}]->state[{self.target_state}] via {self.transition_cursor}"
                )
            else:
                components.append(f"state[{self.current_state}]")
            if self.remaining_input:
                components.append(f"remaining_input={self.remaining_input}")

            return f"WhitespaceAcceptor.Cursor({' | '.join(components)})"
