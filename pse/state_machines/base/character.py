from __future__ import annotations

from collections.abc import Iterable

from pse_core.state_machine import StateMachine
from pse_core.walker import Walker


class CharacterStateMachine(StateMachine):
    """
    Accept multiple characters at once if they are all in the charset.
    Will also prefix the walker with the valid characters if it's not in the
    accepted state.
    """

    def __init__(
        self,
        charset: list[str] | Iterable[str],
        char_min: int | None = None,
        char_limit: int | None = None,
        is_optional: bool = False,
        case_sensitive: bool = True,
    ) -> None:
        """
        Initialize the CharacterStateMachine with a set of valid characters.

        Args:
            charset (list[str]): A list of characters to be accepted.
        """
        super().__init__(is_optional=is_optional, is_case_sensitive=case_sensitive)
        self.char_min = char_min or 0
        self.char_limit = char_limit or 0
        self.charset: set[str] = (
            set(charset) if case_sensitive else set(char.lower() for char in charset)
        )

    def get_new_walker(self, state: int | str) -> CharacterWalker:
        return CharacterWalker(self)

    def __str__(self) -> str:
        sorted_character_set = ", ".join(
            [f"'{char!r}'" for char in sorted(self.charset)]
        )
        return f"{self.__class__.__name__}(charset=[{sorted_character_set}])"


class CharacterWalker(Walker):
    """
    Walker for navigating through characters in CharAcceptor.
    """

    def __init__(
        self,
        state_machine: CharacterStateMachine,
        value: str | None = None,
    ) -> None:
        """
        Initialize the Walker.

        Args:
            state_machine (CharacterStateMachine): The parent CharacterStateMachine.
            value (Optional[str]): The current input value. Defaults to None.
        """
        super().__init__(state_machine)
        self.target_state = "$"
        self.state_machine: CharacterStateMachine = state_machine
        self._raw_value = value
        if value:
            self.consumed_character_count = len(value)

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        """
        Returns a list of valid continuations for the current walker.
        """
        return list(self.state_machine.charset)

    def should_start_transition(self, token: str) -> bool:
        """
        Determines if a transition should start with the given input string.
        """
        token = token.lower() if not self.state_machine.is_case_sensitive else token
        self._accepts_more_input = bool(
            token and token[0] in self.state_machine.charset
        )
        return self._accepts_more_input

    def should_complete_transition(self) -> bool:
        """
        Determines if the transition should be completed based on the character limit.
        """
        if self.state_machine.char_limit > 0:
            return self.consumed_character_count <= self.state_machine.char_limit

        return self.consumed_character_count >= self.state_machine.char_min

    def consume(self, token: str) -> list[Walker]:
        """
        Advance the walker with the given input. Accumulates all valid characters.

        Args:
            token (str): The input to advance with.

        Returns:
            list[Walker]: A list containing the new walker state if input is valid.
        """
        if not token:
            return []

        token = token.lower() if not self.state_machine.is_case_sensitive else token

        # Find valid characters up to char_limit
        valid_length = 0
        for char in token:
            if char not in self.state_machine.charset:
                break
            if (
                self.state_machine.char_limit > 0
                and valid_length + self.consumed_character_count
                >= self.state_machine.char_limit
            ):
                break
            valid_length += 1

        if valid_length == 0:
            return []

        new_value = self.get_raw_value() + token[:valid_length]
        remaining_input = token[valid_length:] if valid_length < len(token) else None

        return self.transition(new_value, remaining_input)

    def can_accept_more_input(self) -> bool:
        """
        Determines if the walker can accept more input based on the character limit.
        """
        if self.state_machine.char_limit > 0:
            return self.consumed_character_count < self.state_machine.char_limit

        return self.consumed_character_count > 0
