from __future__ import annotations

from collections.abc import Iterable

from pse_core.accepted_state import AcceptedState
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker


class CharacterAcceptor(StateMachine):
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
        Initialize the CharAcceptor with a set of valid characters.

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
        return self.__repr__()

    def __repr__(self) -> str:
        sorted_character_set = ", ".join(sorted(self.charset))
        return f"{self.__class__.__name__}(charset=[{sorted_character_set}])"


class CharacterWalker(Walker):
    """
    Walker for navigating through characters in CharAcceptor.
    """

    def __init__(
        self,
        state_machine: CharacterAcceptor,
        value: str | None = None,
    ) -> None:
        """
        Initialize the Walker.

        Args:
            state_machine (CharAcceptor): The parent CharAcceptor.
            value (Optional[str]): The current input value. Defaults to None.
        """
        super().__init__(state_machine)
        self.target_state = "$"
        self.state_machine: CharacterAcceptor = state_machine
        self._raw_value = value

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        return list(self.state_machine.charset)

    def should_start_transition(self, token: str) -> bool:
        """Determines if a transition should start with the given input string."""
        token = token.lower() if not self.state_machine.is_case_sensitive else token
        self._accepts_more_input = bool(
            token and token[0] in self.state_machine.charset
        )
        return self._accepts_more_input

    def consume_token(self, token: str) -> list[Walker]:
        """
        Advance the walker with the given input. Accumulates all valid characters.

        Args:
            token (str): The input to advance with.

        Returns:
            list[Walker]: A list containing the new walker state if input is valid.
        """
        if not token:
            self._accepts_more_input = False
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
            self._accepts_more_input = False
            return []

        # Create new walker with accumulated value
        new_walker = self.__class__(
            self.state_machine, f"{self.get_raw_value()}{token[:valid_length]}"
        )
        new_walker.consumed_character_count = (
            self.consumed_character_count + valid_length
        )
        new_walker.remaining_input = (
            token[valid_length:] if valid_length < len(token) else None
        )
        new_walker._accepts_more_input = not new_walker.remaining_input and (
            self.state_machine.char_limit <= 0
            or valid_length < self.state_machine.char_limit
        )

        return [
            AcceptedState(new_walker)
            if new_walker.consumed_character_count >= self.state_machine.char_min
            else new_walker
        ]

    def get_raw_value(self) -> str:
        return self._raw_value or ""

    def get_current_value(self) -> str | None:
        """
        Retrieve the current value of the walker.

        Returns:
            Optional[str]: The current character or None.
        """
        return self._raw_value

    def is_within_value(self) -> bool:
        """
        Check if the walker has a value.

        Returns:
            bool: True if the walker has a value, False otherwise.
        """
        return self.consumed_character_count > 0
