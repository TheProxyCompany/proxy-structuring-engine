from __future__ import annotations
from typing import Iterable, Optional, Set

from lexpy import DAWG

from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.cursor import Cursor, logger
from pse.state_machine.state_machine import StateMachine

class CharacterAcceptor(StateMachine):
    """
    Accept multiple characters at once if they are all in the charset.
    Will also prefix the cursor with the valid characters if it's not in the
    accepted state.
    """

    def __init__(self, charset: Iterable[str]) -> None:
        """
        Initialize the CharAcceptor with a set of valid characters.

        Args:
            charset (Iterable[str]): An iterable of characters to be accepted.
        """
        super().__init__({}, initial_state=0)
        self.charset: Set[str] = set(charset)

    def __repr__(self) -> str:
        return f"CharAcceptor(charset={self.charset})"

    def get_cursors(self) -> Iterable[Cursor]:
        return [self.Cursor(self)]

    def expects_more_input(self, cursor: Cursor) -> bool:
        return cursor._accepts_remaining_input and cursor.remaining_input is not None

    class Cursor(Cursor):
        """
        Cursor for navigating through characters in CharAcceptor.
        """

        def __init__(
            self, acceptor: CharacterAcceptor, value: Optional[str] = None
        ) -> None:
            """
            Initialize the Cursor.

            Args:
                acceptor (CharAcceptor): The parent CharAcceptor.
                value (Optional[str]): The current input value. Defaults to None.
            """
            super().__init__(acceptor)
            self.acceptor: CharacterAcceptor = acceptor
            self.value: Optional[str] = value

        def select(self, dawg: DAWG) -> Iterable[str]:
            """
            Select characters that are valid based on the acceptor's charset.

            Args:
                candidate_chars (Set[str]): Set of candidate characters (ignored in this implementation).

            Returns:
                Iterable[str]: An iterable of valid characters from the acceptor's charset.
            """
            for char in self.acceptor.charset:
                yield char

        def advance(self, token: str) -> Iterable[Cursor]:
            """
            Advance the cursor with the given input. Accumulates all valid characters.

            Args:
                input (str): The input to advance with.

            Returns:
                List[Cursor]: A list containing the new cursor state if input is valid.
            """
            if not token:
                logger.debug(f"Cursor {self} does not expect more input, returning")
                self._accepts_remaining_input = False
                return

            logger.debug(
                f"Advancing cursor in char acceptor: {self}, with input: '{token}'"
            )

            # Accumulate all valid characters
            valid_chars = ""
            remaining_input = None
            for char in token:
                if char in self.acceptor.charset:
                    valid_chars += char
                else:
                    remaining_input = token[len(valid_chars) :]
                    break

            if valid_chars:
                if self.can_handle_remaining_input:
                    valid_chars = (
                        str(self.value) + valid_chars if self.value else valid_chars
                    )

                new_cursor = self.__class__(self.acceptor, valid_chars)
                new_cursor.remaining_input = remaining_input
                new_cursor.consumed_character_count = (
                    self.consumed_character_count + len(valid_chars)
                )
                logger.debug(f"new_cursor: {AcceptedState(new_cursor)}")
                yield AcceptedState(new_cursor)
            else:
                logger.debug(
                    f"Cursor {self} cannot handle input: {token} and cannot accept remaining input"
                )
                self._accepts_remaining_input = False

        def get_value(self) -> Optional[str]:
            """
            Retrieve the current value of the cursor.

            Returns:
                Optional[str]: The current character or None.
            """
            return self.value

        def is_in_value(self) -> bool:
            """
            Check if the cursor has a value.

            Returns:
                bool: True if the cursor has a value, False otherwise.
            """
            return self.value is not None

        def __repr__(self) -> str:
            """
            Represent the Cursor as a string.

            Returns:
                str: A string representation of the Cursor.
            """
            extra_remaining_input = (
                f" remaining_input={self.remaining_input}"
                if self.remaining_input
                else ""
            )
            return f"CharAcceptor.Cursor({repr(self.acceptor.charset)}, value={self.value}{extra_remaining_input})"


class DigitAcceptor(CharacterAcceptor):
    """
    Accepts one or more digit characters.
    """

    def __init__(self) -> None:
        """
        Initialize the DigitAcceptor with digits 0-9.
        """
        super().__init__("0123456789")

    class Cursor(CharacterAcceptor.Cursor):
        """
        Cursor for navigating through digits in DigitAcceptor.
        """

        def __init__(
            self, acceptor: DigitAcceptor, value: Optional[str] = None
        ) -> None:
            """
            Initialize the Cursor.
            """
            super().__init__(acceptor, value)
            self._accepts_remaining_input = True

        def __repr__(self) -> str:
            return (
                f"DigitAcceptor.Cursor(value={self.value})"
                if self.value
                else "DigitAcceptor.Cursor()"
            )


class HexDigitAcceptor(CharacterAcceptor):
    """
    Accepts one or more hexadecimal digit characters.
    """

    def __init__(self) -> None:
        """
        Initialize the HexDigitAcceptor with hexadecimal digits.
        """
        super().__init__("0123456789ABCDEFabcdef")

# Initialize global instances
digit_acceptor: DigitAcceptor = DigitAcceptor()
hex_digit_acceptor: HexDigitAcceptor = HexDigitAcceptor()
