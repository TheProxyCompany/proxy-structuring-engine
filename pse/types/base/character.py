from __future__ import annotations

from collections.abc import Iterable

from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper


class CharacterStateMachine(StateMachine):
    """
    Accepts one or more valid characters.
    """

    def __init__(
        self,
        charset: str | list[str] | Iterable[str] = "",
        blacklist_charset: str | list[str] | Iterable[str] = "",
        char_min: int | None = None,
        char_limit: int | None = None,
        is_optional: bool = False,
        case_sensitive: bool = True,
    ) -> None:
        """
        Args:
            charset (list[str]): A list of characters to be accepted.
        """
        super().__init__(
            is_optional=is_optional,
            is_case_sensitive=case_sensitive,
        )
        self.char_min = char_min or 0
        self.char_limit = char_limit or 0
        self.charset: set[str] = set()
        self.blacklist_charset: set[str] = set()

        if charset:
            self.charset = (
                set(charset)
                if case_sensitive
                else set(char.lower() for char in charset)
            )
        if blacklist_charset:
            self.blacklist_charset = (
                set(blacklist_charset)
                if case_sensitive
                else set(char.lower() for char in blacklist_charset)
            )

    def get_new_stepper(self, state: int | str) -> CharacterStepper:
        return CharacterStepper(self)

    def __str__(self) -> str:
        return "Character"


class CharacterStepper(Stepper):
    """
    Stepper for navigating through characters in CharacterStateMachine.
    """

    def __init__(
        self,
        state_machine: CharacterStateMachine,
        value: str | None = None,
    ) -> None:
        """
        Initialize the Stepper.

        Args:
            value (Optional[str]): The accumulated string value. Defaults to None.
        """
        super().__init__(state_machine)
        self.target_state = "$"
        self.state_machine: CharacterStateMachine = state_machine
        self._raw_value = value
        if value:
            self.consumed_character_count = len(value)

    def accepts_any_token(self) -> bool:
        return not self.state_machine.charset

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        """
        Returns a list of valid continuations for the current stepper.
        """
        return list(self.state_machine.charset)

    def can_accept_more_input(self) -> bool:
        """
        Determines if the stepper can accept more input based on the character limit.
        """
        if (
            self.state_machine.char_limit > 0
            and self.consumed_character_count >= self.state_machine.char_limit
        ):
            return False

        return True

    def should_start_step(self, token: str) -> bool:
        """
        Determines if a transition should start with the given token.

        Args:
            token (str): The input token to check.

        Returns:
            bool: True if the token can start a transition, False otherwise.
        """
        if not token or (
            self.state_machine.char_limit > 0
            and self.consumed_character_count >= self.state_machine.char_limit
        ):
            return False

        first_char = token[0]
        if not self.state_machine.is_case_sensitive:
            first_char = first_char.lower()

        if first_char in self.state_machine.blacklist_charset:
            return False

        if self.state_machine.charset:
            return first_char in self.state_machine.charset

        return True

    def should_complete_step(self) -> bool:
        """
        Determines if the transition should be completed based on the character limit.
        """
        if (
            self.state_machine.char_limit > 0
            and self.consumed_character_count > self.state_machine.char_limit
        ):
            return False

        if (
            self.state_machine.char_min > 0
            and self.consumed_character_count < self.state_machine.char_min
        ):
            return False

        return True

    def consume(self, token: str) -> list[Stepper]:
        """
        Advance the stepper with the given input.

        Args:
            token (str): The input to advance with.

        Returns:
            List[Stepper]: List of new steppers after advancement.
        """
        if not token or not self.should_start_step(token):
            return []

        token = token.lower() if not self.state_machine.is_case_sensitive else token

        # Split input at first invalid character
        valid_prefix = ""
        for char in token:
            if char in self.state_machine.blacklist_charset or (
                self.state_machine.charset and char not in self.state_machine.charset
            ):
                break
            if (
                self.state_machine.char_limit > 0
                and len(valid_prefix) + self.consumed_character_count
                >= self.state_machine.char_limit
            ):
                break
            valid_prefix += char

        if not valid_prefix:
            return []

        new_value = self.get_raw_value() + valid_prefix
        remaining_input = token[len(valid_prefix) :] or None
        new_stepper = self.step(new_value, remaining_input)

        return [new_stepper]
