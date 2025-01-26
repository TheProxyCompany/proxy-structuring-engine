from __future__ import annotations

import logging

from pse.state_machines.base.character import CharacterStateMachine

logger = logging.getLogger()

WHITESPACE_CHARS: str = " \n\r\t"


class WhitespaceStateMachine(CharacterStateMachine):
    """
    Optional whitespace state_machine using TokenTrie for efficient matching.
    """

    def __init__(self, min_whitespace: int = 0, max_whitespace: int = 40):
        """
        Args:
            min_whitespace (int, optional): Minimum allowable whitespace characters.
                Defaults to 0.
            max_whitespace (int, optional): Maximum allowable whitespace characters.
                Defaults to 40.
        """
        super().__init__(
            WHITESPACE_CHARS,
            min_whitespace,
            max_whitespace,
            is_optional=(min_whitespace == 0),
        )

    def __str__(self) -> str:
        return "Whitespace"
