from __future__ import annotations

import logging

from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

logger = logging.getLogger(__name__)


class PhraseStateMachine(StateMachine):
    """
    Accepts a predefined sequence of characters, validating input against the specified text.

    Attributes:
        phrase (str): The target string that this state_machine is validating against.
    """

    def __init__(
        self,
        phrase: str,
        is_optional: bool = False,
        is_case_sensitive: bool = True,
    ):
        """
        Initialize a new PhraseStateMachine instance with the specified text.

        Args:
            phrase (str): The string of characters that this state_machine will validate.
                Must be a non-empty string.

        Raises:
            ValueError: If the provided text is empty.
        """
        super().__init__(is_optional=is_optional, is_case_sensitive=is_case_sensitive)

        if not phrase:
            raise ValueError("Phrase must be a non-empty string.")

        self.phrase = phrase

    def get_new_stepper(self, state: int | str | None = None) -> PhraseStepper:
        return PhraseStepper(self)

    def __str__(self) -> str:
        """
        Provide a string representation of the PhraseStateMachine.

        Returns:
            str: A string representation of the PhraseStateMachine.
        """
        return f"Phrase({self.phrase!r})"


class PhraseStepper(Stepper):
    def __init__(
        self,
        state_machine: PhraseStateMachine,
        consumed_character_count: int | None = None,
    ):
        super().__init__(state_machine)
        if consumed_character_count is not None and consumed_character_count < 0:
            raise ValueError("Consumed character count must be non-negative")

        self.consumed_character_count = consumed_character_count or 0
        self.state_machine: PhraseStateMachine = state_machine
        self.target_state = "$"

    def can_accept_more_input(self) -> bool:
        """
        Check if the stepper can accept more input.
        """
        return self.consumed_character_count < len(self.state_machine.phrase)

    def should_start_step(self, token: str) -> bool:
        """
        Start a transition if the token is not empty and matches the remaining text.
        """
        if not token:
            return False

        valid_length = self._get_valid_match_length(token)
        return valid_length > 0

    def should_complete_step(self) -> bool:
        return self.consumed_character_count == len(self.state_machine.phrase)

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        if self.consumed_character_count >= len(self.state_machine.phrase):
            return []

        remaining_text = self.state_machine.phrase[self.consumed_character_count :]
        return [remaining_text]

    def consume(self, token: str) -> list[Stepper]:
        """
        Advances the stepper if the token matches the expected text at the current position.
        Args:
            token (str): The string to match against the expected text.

        Returns:
            list[Stepper]: A stepper if the token matches, empty otherwise.
        """
        valid_length = self._get_valid_match_length(token)
        if valid_length <= 0:
            return []

        new_value = self.get_raw_value() + token[:valid_length]
        remaining_input = token[valid_length:] if valid_length < len(token) else None
        new_stepper = self.step(new_value, remaining_input)
        return [new_stepper]

    def get_raw_value(self) -> str:
        return self.state_machine.phrase[: self.consumed_character_count]

    def _get_valid_match_length(self, token: str, pos: int | None = None) -> int:
        """
        Calculate the length of the matching prefix between the token and the target phrase.
        
        Args:
            token: The input string to check against the target phrase
            pos: Starting position in the phrase (defaults to current consumed count)
            
        Returns:
            Length of the matching prefix
        """
        # Use current position if not specified
        pos = pos or self.consumed_character_count
        
        # Get the remaining portion of the phrase to match against
        remaining_phrase = self.state_machine.phrase[pos:]
        
        # Determine maximum possible match length
        max_length = min(len(token), len(remaining_phrase))
        
        # Find the longest matching prefix using string slicing
        # This is more efficient than character-by-character comparison
        for i in range(max_length + 1):
            if token[:i] != remaining_phrase[:i]:
                return i - 1
        
        return max_length

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PhraseStepper)
            and super().__eq__(other)
            and self.state_machine.phrase == other.state_machine.phrase
        )
