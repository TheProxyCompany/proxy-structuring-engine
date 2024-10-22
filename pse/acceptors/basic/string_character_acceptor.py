from __future__ import annotations
from typing import Iterable, Optional, Type, Set

from lexpy import DAWG

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.cursor import Cursor, logger

# INVALID_CHARS is a set containing characters that are not allowed in JSON strings.
# It includes control characters (ASCII 0-31) and the double quote (") and backslash (\) characters.
# These characters need to be escaped or are not permitted in JSON string values.
INVALID_CHARS: Set[str] = {chr(c) for c in range(0, 0x20)} | {'"', "\\"}

class StringCharacterAcceptor(TokenAcceptor):
    """
    Accepts one or more valid JSON unescaped string characters.
    """

    def __init__(self) -> None:
        """
        Initialize the StringCharAcceptor with its state transitions.
        """
        super().__init__(initial_state=0, end_states={1})

    @property
    def cursor_class(self) -> Type[Cursor]:
        return StringCharacterAcceptor.Cursor

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

    def advance_cursor(self, cursor: Cursor, input: str) -> Iterable[Cursor]:
        old_value = cursor.get_value()
        new_value = old_value + input if old_value else input
        new_cursor: StringCharacterAcceptor.Cursor = StringCharacterAcceptor.Cursor(
            self, new_value
        )
        yield AcceptedState(new_cursor)

    def expects_more_input(self, cursor: Cursor) -> bool:
        """
        StringCharAcceptor assumes it doesn't expect more input on its own.
        It's controlled by the parent acceptor (StringAcceptor).

        Args:
            cursor (Cursor): The current cursor.

        Returns:
            bool: False
        """
        return False

    class Cursor(Cursor):
        """
        Cursor for navigating through characters in StringCharAcceptor.
        """

        def __init__(
            self, acceptor: StringCharacterAcceptor, value: Optional[str] = None
        ) -> None:
            """
            Initialize the Cursor.

            Args:
                acceptor (StringCharAcceptor): The parent StringCharAcceptor.
                value (Optional[str]): The accumulated string value. Defaults to None.
            """
            super().__init__(acceptor)
            self.acceptor: StringCharacterAcceptor = acceptor
            self.value: Optional[str] = value
            self._accepts_remaining_input = True

        def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
            """
            Select valid string characters by excluding invalid ones.

            Returns:
                Set[str]: Set of valid string characters.
            """
            return self.acceptor.valid_chars

        def advance(self, token: str) -> Iterable[Cursor]:
            """
            Advance the cursor with the given input.

            Args:
                input (str): The input to advance with.

            Returns:
                List[Cursor]: List of new cursors after advancement.
            """
            logger.debug(
                f"Advancing cursor in string char acceptor: {self}, with input: {token}"
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
                new_cursor = self.clone()
                new_cursor.value = (self.value if self.value else "") + valid_prefix
                new_cursor.remaining_input = remaining_input
                new_cursor.consumed_character_count += len(valid_prefix)
                logger.debug(
                    f"Valid prefix: {valid_prefix}, Remaining input: {remaining_input}"
                )
                yield AcceptedState(new_cursor)
            else:
                self._accepts_remaining_input = False

        def get_value(self) -> Optional[str]:
            """
            Retrieve the accumulated string value.

            Returns:
                Optional[str]: The accumulated string or None.
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
            Return an unambiguous string representation of the instance.

            Returns:
                str: The string representation including value and acceptor.

            Example:
                StringCharacterAcceptor.Cursor("valid_value | remaining_input='abc' | state[0]")
            """
            components = []
            if self.get_value():
                components.append(f"value='{self.get_value()}'")
            if self.remaining_input:
                components.append(f"remaining_input='{self.remaining_input}'")

            return f"String.Cursor({' | '.join(components)})"
