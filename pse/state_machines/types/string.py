from __future__ import annotations

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.base.any_character import AnyCharacterStateMachine
from pse.state_machines.base.character import CharacterStateMachine
from pse.state_machines.base.phrase import PhraseStateMachine

INVALID_CHARS: set[str] = {chr(c) for c in range(0, 0x20)} | {'"', "\\"}


class StringStateMachine(StateMachine):
    """
    Accepts a well-formed JSON string.

    The length of the string is measured excluding the surrounding quotation marks.
    """

    # StateId constants
    STRING_CONTENTS = 1
    ESCAPED_SEQUENCE = 2
    HEX_CODE = 3

    def __init__(self):
        """
        The state machine is configured to parse JSON strings, handling escape sequences
        and Unicode characters appropriately.
        """
        super().__init__(
            {
                0: [
                    (PhraseStateMachine('"'), self.STRING_CONTENTS),
                ],
                self.STRING_CONTENTS: [
                    (
                        AnyCharacterStateMachine(disallowed_charset=INVALID_CHARS),
                        self.STRING_CONTENTS,
                    ),  # Regular characters
                    (PhraseStateMachine('"'), "$"),  # End quote
                    (
                        PhraseStateMachine("\\"),
                        self.ESCAPED_SEQUENCE,
                    ),  # Escape character
                ],
                self.ESCAPED_SEQUENCE: [
                    (
                        CharacterStateMachine('"\\/bfnrt', char_limit=1),
                        self.STRING_CONTENTS,
                    ),  # Escaped characters
                    (PhraseStateMachine("u"), self.HEX_CODE),  # Unicode escape sequence
                ],
                self.HEX_CODE: [
                    (
                        CharacterStateMachine(
                            "0123456789ABCDEFabcdef",
                            char_min=4,
                            char_limit=4,
                        ),
                        self.STRING_CONTENTS,
                    ),
                ],
            }
        )

    def get_new_stepper(self, state: int | str | None = None) -> Stepper:
        return StringStepper(self, state)

    def __str__(self) -> str:
        return "String"


class StringStepper(Stepper):
    def __init__(
        self, state_machine: StringStateMachine, current_state: StateId | None = None
    ) -> None:
        super().__init__(state_machine, current_state)
        self.state_machine: StringStateMachine = state_machine

    def is_within_value(self) -> bool:
        return self.current_state == 1
