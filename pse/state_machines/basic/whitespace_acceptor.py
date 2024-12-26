from __future__ import annotations

import logging

from pse_core.accepted_state import AcceptedState
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

logger = logging.getLogger()

WHITESPACE_CHARS: str = " \n\r\t"


class WhitespaceAcceptor(StateMachine):
    """
    Optional whitespace state_machine using TokenTrie for efficient matching.
    """

    def __init__(self, min_whitespace: int = 0, max_whitespace: int = 40):
        """
        Initialize the WhitespaceAcceptor.

        Args:
            min_whitespace (int, optional): Minimum allowable whitespace characters.
                Defaults to 0.
            max_whitespace (int, optional): Maximum allowable whitespace characters.
                Defaults to 40.
        """
        self.min_whitespace: int = min_whitespace
        self.max_whitespace: int = max_whitespace

        super().__init__(is_optional=(min_whitespace == 0))

    def get_new_walker(self, state: int | str) -> WhitespaceWalker:
        return WhitespaceWalker(self)


class WhitespaceWalker(Walker):
    """
    Walker for WhitespaceAcceptor utilizing TokenTrie.
    """

    def __init__(
        self,
        state_machine: WhitespaceAcceptor,
        value: str | None = None,
    ):
        """
        Initialize the walker.

        Args:
            state_machine (WhitespaceAcceptor): The parent state_machine.
            text (str, optional): Accumulated whitespace text. Defaults to "".
        """
        super().__init__(state_machine)
        self.target_state = "$"
        self.state_machine: WhitespaceAcceptor = state_machine
        self._raw_value = value or ""
        self.consumed_character_count = len(self._raw_value)
        self.length_exceeded: bool = (
            len(self._raw_value) > self.state_machine.max_whitespace
        )

    def should_start_transition(self, token: str) -> bool:
        if not token or not token[0].isspace():
            self._accepts_more_input = False
            return False

        return True

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        if self.length_exceeded:
            return []

        return list(WHITESPACE_CHARS)

    def can_accept_more_input(self) -> bool:
        return (
            self._accepts_more_input
            and self.consumed_character_count < self.state_machine.max_whitespace
        )

    def consume_token(self, token: str) -> list[Walker]:
        """
        Advance the walker with the given characters.
        Args:
            input (str): The characters to advance with.

        Returns:
            List[WhitespaceAcceptor.Walker]: List of new walkers after advancement.
        """
        if self.length_exceeded:
            return []

        # Extract the valid whitespace prefix
        valid_length = 0
        if token.isspace():
            valid_length = len(token)
        else:
            for c in token:
                if c.isspace():
                    valid_length += 1
                else:
                    break

        valid_input = token[:valid_length]
        remaining_input = token[valid_length:] or None

        if not valid_input:
            self._accepts_more_input = False
            if (
                remaining_input
                and self.consumed_character_count >= self.state_machine.min_whitespace
            ):
                copy = self.clone()
                copy._raw_value = (self._raw_value or "") + valid_input
                copy.remaining_input = remaining_input
                copy._accepts_more_input = False
                return [AcceptedState(copy)]
            return []

        next_walker = self.clone()
        next_walker._raw_value = (self._raw_value or "") + valid_input
        next_walker.remaining_input = remaining_input
        next_walker.consumed_character_count += valid_length

        if next_walker.consumed_character_count > self.state_machine.max_whitespace:
            return []

        next_walker._accepts_more_input = (
            remaining_input is None
            and next_walker.consumed_character_count
            <= self.state_machine.max_whitespace
        )

        return [
            AcceptedState(next_walker)
            if next_walker.consumed_character_count >= self.state_machine.min_whitespace
            else next_walker
        ]

    def get_current_value(self) -> str:
        """
        Get the accumulated whitespace value.

        Returns:
            str: The whitespace string.
        """
        return self.get_raw_value() or ""

    def is_within_value(self) -> bool:
        return self.consumed_character_count > 0
