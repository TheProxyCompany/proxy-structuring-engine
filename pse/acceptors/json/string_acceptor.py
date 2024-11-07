from __future__ import annotations
import json
from typing import Optional, Type
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineWalker,
    StateMachineGraph,
)
from pse.acceptors.basic.character_acceptors import (
    CharacterAcceptor,
    hex_digit_acceptor,
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
    STATE_UNICODE_HEX_1 = 3
    STATE_UNICODE_HEX_2 = 4
    STATE_UNICODE_HEX_3 = 5
    STATE_UNICODE_HEX_4 = 6

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
                    CharacterAcceptor('"\\/bfnrt'),
                    self.STATE_IN_STRING,
                ),  # Escaped characters
                (
                    TextAcceptor("u"),
                    self.STATE_UNICODE_HEX_1,
                ),  # Unicode escape sequence
            ],
            self.STATE_UNICODE_HEX_1: [
                (hex_digit_acceptor, self.STATE_UNICODE_HEX_2),  # First hex digit
            ],
            self.STATE_UNICODE_HEX_2: [
                (hex_digit_acceptor, self.STATE_UNICODE_HEX_3),  # Second hex digit
            ],
            self.STATE_UNICODE_HEX_3: [
                (hex_digit_acceptor, self.STATE_UNICODE_HEX_4),  # Third hex digit
            ],
            self.STATE_UNICODE_HEX_4: [
                (
                    hex_digit_acceptor,
                    self.STATE_IN_STRING,
                ),  # Fourth hex digit, return to state IN_STRING
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

    def should_complete_transition(
        self, transition_value: str, is_end_state: bool
    ) -> bool:
        """
        Handle the completion of a transition.

        Args:
            transition_value (str): The value transitioned with.
            target_state (Any): The target state after transition.
            is_end_state (bool): Indicates if the transition leads to an end state.

        Returns:
            bool: Success of the transition.
        """
        if is_end_state and self.raw_value:
            try:
                self.value = json.loads(self.raw_value)
            except json.JSONDecodeError:
                self._accepts_remaining_input = False
                return False

        return True
