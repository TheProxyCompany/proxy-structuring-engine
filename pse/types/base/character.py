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
        whitelist_charset: str | list[str] | Iterable[str] = "",
        graylist_charset: str | list[str] | Iterable[str] = "",
        blacklist_charset: str | list[str] | Iterable[str] = "",
        char_min: int | None = None,
        char_limit: int | None = None,
        is_optional: bool = False,
        case_sensitive: bool = True,
    ) -> None:
        """
        Initialize a CharacterStateMachine with character sets and constraints.
        
        Args:
            whitelist_charset: Characters that are explicitly allowed
            graylist_charset: Characters that are allowed but terminate the match if they follow other characters
            blacklist_charset: Characters that are explicitly forbidden
            char_min: Minimum number of characters required (0 if None)
            char_limit: Maximum number of characters allowed (unlimited if 0 or None)
            is_optional: Whether this state machine is optional
            case_sensitive: Whether character matching is case-sensitive
        """
        super().__init__(
            is_optional=is_optional,
            is_case_sensitive=case_sensitive,
        )
        self.char_min = char_min or 0
        self.char_limit = char_limit or 0
        self.charset: set[str] = set()
        self.graylist_charset: set[str] = set()
        self.blacklist_charset: set[str] = set()

        # Process all charsets efficiently
        def convert_to_set(chars):
            return set(char.lower() for char in chars) if chars else set()

        self.charset = set(whitelist_charset) if case_sensitive else convert_to_set(whitelist_charset)
        self.graylist_charset = set(graylist_charset) if case_sensitive else convert_to_set(graylist_charset)
        self.blacklist_charset = set(blacklist_charset) if case_sensitive else convert_to_set(blacklist_charset)

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

        This method processes the input token and determines how much of it can be consumed
        based on character constraints. It stops consuming at the first invalid character.

        Args:
            token: The input string to consume

        Returns:
            List of new steppers after advancement (empty if nothing can be consumed)
        """
        if not token or not self.should_start_step(token):
            return []

        # Apply case sensitivity
        token = token.lower() if not self.state_machine.is_case_sensitive else token

        # Cache frequently used properties for performance
        charset = self.state_machine.charset
        blacklist = self.state_machine.blacklist_charset
        graylist = self.state_machine.graylist_charset
        char_limit = self.state_machine.char_limit
        consumed_count = self.consumed_character_count

        # Find the longest valid prefix efficiently
        valid_prefix_len = 0
        for char in token:
            # Stop at first invalid character or limit
            is_blacklisted = char in blacklist
            not_in_charset = len(charset) > 0 and char not in charset
            exceeds_limit = char_limit > 0 and valid_prefix_len + consumed_count >= char_limit
            is_graylisted = len(graylist) > 0 and valid_prefix_len > 0 and char in graylist

            if is_blacklisted or not_in_charset or exceeds_limit or is_graylisted:
                break

            valid_prefix_len += 1

        # Extract the valid portion using string slicing
        valid_prefix = token[:valid_prefix_len]

        # Create new stepper with updated state
        new_value = self.get_raw_value() + valid_prefix
        remaining_input = token[len(valid_prefix):] or None
        new_stepper = self.step(new_value, remaining_input)

        return [new_stepper]
