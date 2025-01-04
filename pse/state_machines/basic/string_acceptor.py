from __future__ import annotations

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.basic.character_acceptor import CharacterAcceptor
from pse.state_machines.basic.string_character_acceptor import StringCharacterAcceptor
from pse.state_machines.basic.text_acceptor import TextAcceptor


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
                    (StringCharacterAcceptor(), self.STRING_CONTENTS),  # Regular characters
                    (TextAcceptor('"'), "$"),  # End quote
                    (TextAcceptor("\\"), self.ESCAPED_SEQUENCE),  # Escape character
                ],
                self.ESCAPED_SEQUENCE: [
                    (CharacterAcceptor('"\\/bfnrt', char_limit=1), self.STRING_CONTENTS),  # Escaped characters
                    (TextAcceptor("u"), self.HEX_CODE),  # Unicode escape sequence
                ],
                self.HEX_CODE: [
                    (CharacterAcceptor(
                        "0123456789ABCDEFabcdef",
                        char_min=4,
                        char_limit=4,
                    ), self.STRING_CONTENTS),
                ],
            }
        )

    def get_new_walker(self, state: State) -> StringWalker:
        return StringWalker(self, state)


class StringWalker(Walker):
    """
    Walker for StringAcceptor.

    Manages the parsing state and accumulates characters for a JSON string.
    The length attribute tracks the number of characters in the string content,
    explicitly excluding the opening and closing quotation marks.
    """

    def __init__(
        self, state_machine: StringAcceptor, current_state: State | None = None
    ):
        """
        Initialize the walker.

        Args:
            state_machine (StringAcceptor): The parent state_machine.
        """
        super().__init__(state_machine, current_state)
        self.state_machine: StringAcceptor = state_machine
