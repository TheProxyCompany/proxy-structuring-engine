from __future__ import annotations

from lexpy import DAWG

from pse.state_machine.state_machine import StateMachine
from pse.state_machine.cursor import Cursor
from pse.state_machine.accepted_state import AcceptedState
from typing import Iterable, Set
import logging

logger = logging.getLogger(__name__)

class TextAcceptor(StateMachine):
    """
    Accepts a predefined sequence of characters, validating input against the specified text.

    Attributes:
        text (str): The target string that this acceptor is validating against.
    """

    def __init__(self, text: str):
        """
        Initialize a new TextAcceptor instance with the specified text.

        Args:
            text (str): The string of characters that this acceptor will validate.
                Must be a non-empty string.

        Raises:
            ValueError: If the provided text is empty.
        """
        super().__init__({})
        if not text:
            raise ValueError("TextAcceptor requires a non-empty string.")

        self.text = text

    def get_cursors(self) -> Iterable[Cursor]:
        """
        Get one or more cursors to traverse the acceptor.
        Override.
        """
        return [self.__class__.Cursor(self)]

    def __repr__(self) -> str:
        """
        Provide a string representation of the TextAcceptor.

        Returns:
            str: A string representation of the TextAcceptor.
        """
        return f"TextAcceptor({repr(self.text)})"

    class Cursor(StateMachine.Cursor):
        """
        Represents the current position within the TextAcceptor's text during validation.

        Attributes:
            acceptor (TextAcceptor): The TextAcceptor instance that owns this Cursor.
            consumed_character_count (int): The current position in the text being validated.
        """

        def __init__(self, acceptor: TextAcceptor, consumed_character_count: int = 0):
            """
            Initialize a new Cursor instance.

            Args:
                acceptor (TextAcceptor): The TextAcceptor instance associated with this cursor.
                consumed_character_count (int, optional): The initial position in the text. Defaults to 0.
            """
            super().__init__(acceptor)
            self.acceptor = acceptor
            self.consumed_character_count = consumed_character_count
            self._accepts_remaining_input = True

        def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
            """
            Selects exact partial matches of the target text from the current position.

            Args:
                dawg (DAWG): The DAWG to select from.
                depth (int): The current depth of the cursor in the state machine.
            Returns:
                Set[str]: A set containing exact partial matches of the target text.
            """
            if self.consumed_character_count >= len(self.acceptor.text):
                return set()

            remaining_text = self.acceptor.text[self.consumed_character_count:]
            results = set()

            # Only check if the exact partial text exists in the DAWG
            if remaining_text in dawg:
                results.add(remaining_text)


            # Check if the exact partial prefixes exist in the DAWG
            max_possible_match_len = min(len(remaining_text), 8)
            for i in range(1, max_possible_match_len):
                partial = remaining_text[:i]
                if partial in dawg:
                    results.add(partial)

            return results

        def advance(self, value: str) -> Iterable[Cursor]:
            """
            Advances the cursor if the given value matches the expected text at the current position.
            Args:
                value (str): The string to match against the expected text.

            Returns:
                Iterable[Cursor]: A list containing the next cursor if the value matches,
                                  or an empty list otherwise.
            """
            logger.debug(f"advancing cursor: {self}, value: {value}, self.consumed_character_count: {self.consumed_character_count}")
            expected_text = self.acceptor.text
            pos = self.consumed_character_count

            max_possible_match_len = len(expected_text) - pos
            input_len = len(value)
            match_len = min(max_possible_match_len, input_len)

            # Get the segment to compare
            expected_segment = expected_text[pos:pos + match_len]
            input_segment = value[:match_len]

            logger.debug(f"expected_segment: {expected_segment}, input_segment: {input_segment}")

            if expected_segment == input_segment:
                new_pos = pos + match_len
                remaining_input = value[match_len:]
                if remaining_input:
                    logger.debug(f"remaining_input: {remaining_input}")

                next_cursor = self.__class__(self.acceptor, new_pos)
                next_cursor.remaining_input = remaining_input if remaining_input else None
                if new_pos == len(expected_text):
                    logger.debug(f"found match for {expected_text}, yielding {AcceptedState(next_cursor)}")
                    yield AcceptedState(next_cursor)
                else:
                    yield next_cursor

        def get_value(self) -> str:
            """
            Retrieves the current state of the text being accepted, highlighting the remaining portion.

            Returns:
                str: The accepted portion of the text followed by a marker and the remaining text,
                     e.g., 'hel👉lo' if consumed_character_count is 3.
            """
            return (
                f"{self.acceptor.text[:self.consumed_character_count]}👉{self.acceptor.text[self.consumed_character_count:]}"
                if self.consumed_character_count < len(self.acceptor.text)
                else self.acceptor.text
            )

        def is_in_value(self) -> bool:
            """
            Determine if the cursor is currently within a value.

            Returns:
                bool: True if in a value, False otherwise.
            """
            return self.consumed_character_count > 0 and self.consumed_character_count < len(self.acceptor.text)

        def __repr__(self) -> str:
            """
            Provide a string representation of the Cursor.

            Returns:
                str: A string representation of the Cursor.
            """
            value = (
                f"{self.acceptor.text[:self.consumed_character_count]}👉{self.acceptor.text[self.consumed_character_count:]}"
                if self.consumed_character_count < len(self.acceptor.text)
                else self.acceptor.text
            )
            if self.consumed_character_count == len(self.acceptor.text):
                return f"{self.acceptor}"
            else:
                return f"{self.acceptor.__class__.__name__}.Cursor(value=`{value}`)"

        @property
        def can_handle_remaining_input(self) -> bool:
            """
            Indicates if the cursor can handle more input.

            Returns:
                bool: True if more input can be handled, False otherwise.
            """
            # If not all characters have been consumed, can handle more input
            return self._accepts_remaining_input and self.consumed_character_count < len(self.acceptor.text)
