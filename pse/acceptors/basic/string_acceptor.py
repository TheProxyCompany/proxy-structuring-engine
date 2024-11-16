from __future__ import annotations
from typing import Type
from pse.core.state_machine import (
    StateMachine,
    StateMachineWalker,
)
from pse.acceptors.basic.character_acceptor import CharacterAcceptor
from pse.core.walker import Walker
from pse.acceptors.basic.string_character_acceptor import StringCharacterAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor


class StringAcceptor(StateMachine):
    """
    Accepts a well-formed JSON string.

    The length of the string is measured excluding the surrounding quotation marks.
    """

    # State constants
    STRING_CONTENTS = 1
    ESCAPED_SEQUENCE = 2
    HEX_CODE = 3

    def __init__(self):
        """
        Initialize the StringAcceptor with its state transitions.

        The state machine is configured to parse JSON strings, handling escape sequences
        and Unicode characters appropriately.
        """
        super().__init__(
            {
                0: [
                    (TextAcceptor('"'), self.STRING_CONTENTS),
                ],
                self.STRING_CONTENTS: [
                    (
                        StringCharacterAcceptor(),
                        self.STRING_CONTENTS,
                    ),  # Regular chars first
                    (TextAcceptor('"'), "$"),  # End quote second
                    (TextAcceptor("\\"), self.ESCAPED_SEQUENCE),  # Escape last
                ],
                self.ESCAPED_SEQUENCE: [
                    (
                        CharacterAcceptor('"\\/bfnrt', char_limit=1),
                        self.STRING_CONTENTS,
                    ),  # Escaped characters
                    (
                        TextAcceptor("u"),
                        self.HEX_CODE,
                    ),  # Unicode escape sequence
                ],
                self.HEX_CODE: [
                    (
                        CharacterAcceptor(
                            "0123456789ABCDEFabcdef", char_min=4, char_limit=4
                        ),
                        self.STRING_CONTENTS,
                    ),  # First hex digit
                ],
            }
        )

    @property
    def walker_class(self) -> Type[Walker]:
        return StringWalker


class StringWalker(StateMachineWalker):
    """
    Walker for StringAcceptor.

    Manages the parsing state and accumulates characters for a JSON string.
    The length attribute tracks the number of characters in the string content,
    explicitly excluding the opening and closing quotation marks.
    """

    MAX_LENGTH = 10000  # Define a maximum allowed string length

    def __init__(self, acceptor: StringAcceptor, current_state: int = 0):
        """
        Initialize the walker.

        Args:
            acceptor (StringAcceptor): The parent acceptor.
        """
        super().__init__(acceptor, current_state)
        self.acceptor = acceptor
