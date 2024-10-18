from __future__ import annotations
import json
from typing import (
    Optional,
    Any,
)

from pse.state_machine.state_machine import StateMachine, StateMachineGraph
from pse.acceptors.basic.character_acceptors import CharacterAcceptor, hex_digit_acceptor
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

    def __init__(self):
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
                (TextAcceptor('"'), "$"),  # End of string
                (TextAcceptor("\\"), self.STATE_ESCAPE),  # Escape character detected
                (
                    StringCharacterAcceptor(),
                    self.STATE_IN_STRING,
                ),  # Regular string character
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
        super().__init__(graph)

    # def expects_more_input(self, cursor: StringAcceptor.Cursor) -> bool:
    #     return cursor.current_state == self.STATE_IN_STRING or cursor.current_state == self.initial_state

    class Cursor(StateMachine.Cursor):
        """
        Cursor for StringAcceptor.

        Manages the parsing state and accumulates characters for a JSON string.
        The length attribute tracks the number of characters in the string content,
        explicitly excluding the opening and closing quotation marks.
        """

        MAX_LENGTH = 10000  # Define a maximum allowed string length

        def __init__(self, acceptor: StringAcceptor):
            """
            Initialize the cursor.

            Args:
                acceptor (StringAcceptor): The parent acceptor.
            """
            super().__init__(acceptor)
            self.acceptor = acceptor
            self.text: str = ""
            self.value: Optional[str] = None
            self._accepts_remaining_input = True

        def complete_transition(
            self, transition_value: str, target_state: Any, is_end_state: bool
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
            self.text += transition_value
            self.current_state = target_state
            self._accepts_remaining_input = False
            if is_end_state:
                try:
                    self.value = json.loads(self.text)
                except json.JSONDecodeError:
                    self.value = None
            else:
                self.value = None
            return True

        def get_value(self) -> str:
            """
            Get the accumulated string value.

            Returns:
                str: The parsed string value without surrounding quotes if fully parsed,
                     otherwise the current accumulated text.
            """
            return self.value if self.value is not None else self.text

        def __repr__(self) -> str:
            """
            Return an unambiguous string representation of the instance.

            Returns:
                The string representation including value and acceptor.

            Example:
                StringAcceptor.Cursor(" | transition_cursor=TextAcceptor.Cursor('👉"') | state[1]->state[$])
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

            return f"StringAcceptor.Cursor({' | '.join(components)})"
