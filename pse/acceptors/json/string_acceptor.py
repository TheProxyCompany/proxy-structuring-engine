from __future__ import annotations
from typing import Optional, Type
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineWalker,
    StateMachineGraph,
)
from pse.acceptors.basic.character_acceptor import (
    CharacterAcceptor,
)
from pse.state_machine.walker import Walker
from pse.acceptors.basic.string_character_acceptor import StringCharacterAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor


class StringAcceptor(StateMachine):
    """
    Accepts a well-formed JSON string.

    The length of the string is measured excluding the surrounding quotation marks.
    """

    # State constants
    STATE_START = 0
    STATE_IN_STRING = 1
    STATE_ESCAPE = 2
    STATE_UNICODE_HEX = 3

    def __init__(self, walker_type: Optional[Type[Walker]] = None):
        """
        Initialize the StringAcceptor with its state transitions.

        The state machine is configured to parse JSON strings, handling escape sequences
        and Unicode characters appropriately.
        """
        graph: StateMachineGraph = {
            self.STATE_START: [
                (TextAcceptor('"'), self.STATE_IN_STRING),  # Start of string
            ],
            self.STATE_IN_STRING: [
                (
                    StringCharacterAcceptor(),
                    self.STATE_IN_STRING,
                ),  # Regular chars first
                (TextAcceptor('"'), "$"),  # End quote second
                (TextAcceptor("\\"), self.STATE_ESCAPE),  # Escape last
            ],
            self.STATE_ESCAPE: [
                (
                    CharacterAcceptor('"\\/bfnrt', char_limit=1),
                    self.STATE_IN_STRING,
                ),  # Escaped characters
                (
                    TextAcceptor("u"),
                    self.STATE_UNICODE_HEX,
                ),  # Unicode escape sequence
            ],
            self.STATE_UNICODE_HEX: [
                (
                    CharacterAcceptor("0123456789ABCDEFabcdef", char_min=4, char_limit=4),
                    self.STATE_IN_STRING,
                ),  # First hex digit
            ],
        }
        super().__init__(
            graph,
            self.STATE_START,
            ["$"],
            walker_type=walker_type or StringWalker,
        )


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
