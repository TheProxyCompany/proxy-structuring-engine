from __future__ import annotations

import logging

from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

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

    def get_new_walker(self, state: int | str | None = None) -> PhraseWalker:
        return PhraseWalker(self)

    def __str__(self) -> str:
        """
        Provide a string representation of the PhraseStateMachine.

        Returns:
            str: A string representation of the PhraseStateMachine.
        """
        return f"Phrase({self.phrase!r})"


class PhraseWalker(Walker):
    """
    Represents the current position within the TextAcceptor's text during validation.

    Attributes:
        state_machine (TextAcceptor): The TextAcceptor instance that owns this Walker.
        consumed_character_count (int): The current position in the text being validated.
    """

    def __init__(
        self,
        state_machine: PhraseStateMachine,
        consumed_character_count: int | None = None,
    ):
        """
        Initialize a new Walker instance.

        Args:
            state_machine (TextAcceptor): The TextAcceptor instance associated with this walker.
            consumed_character_count (int, optional): The initial position in the text. Defaults to 0.
        """
        super().__init__(state_machine)
        if consumed_character_count is not None and consumed_character_count < 0:
            raise ValueError("Consumed character count must be non-negative")

        self.consumed_character_count = consumed_character_count or 0
        self.state_machine: PhraseStateMachine = state_machine
        self.target_state = "$"

    def can_accept_more_input(self) -> bool:
        """
        Check if the walker can accept more input.
        """
        return self.consumed_character_count < len(self.state_machine.phrase)

    def should_start_transition(self, token: str) -> bool:
        """
        Start a transition if the token is not empty and matches the remaining text.
        """
        if not token:
            return False

        valid_length = self._get_valid_match_length(token)
        return valid_length > 0

    def should_complete_transition(self) -> bool:
        return self.consumed_character_count == len(self.state_machine.phrase)

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        if self.consumed_character_count >= len(self.state_machine.phrase):
            return []

        valid_continuations = []

        remaining_text = self.state_machine.phrase[self.consumed_character_count :]
        valid_continuations.append(remaining_text)

        # Check if the exact partial prefixes exist in the DAWG
        max_possible_match_len = len(remaining_text)
        for i in range(1, max_possible_match_len):
            partial = remaining_text[:i]
            valid_continuations.append(partial)

        return valid_continuations

    def consume(self, token: str) -> list[Walker]:
        """
        Advances the walker if the token matches the expected text at the current position.
        Args:
            token (str): The string to match against the expected text.

        Returns:
            list[Walker]: A walker if the token matches, empty otherwise.
        """
        valid_length = self._get_valid_match_length(token)
        if valid_length <= 0:
            return []

        new_value = self.get_raw_value() + token[:valid_length]
        remaining_input = token[valid_length:] if valid_length < len(token) else None
        new_walker = self.transition(new_value, remaining_input)
        return [new_walker]

    def get_raw_value(self) -> str:
        return self.state_machine.phrase[: self.consumed_character_count]

    def _get_valid_match_length(self, token: str, pos: int | None = None) -> int:
        pos = pos or self.consumed_character_count
        valid_length = 0
        for i, c in enumerate(token):
            if (
                pos + i < len(self.state_machine.phrase)
                and c == self.state_machine.phrase[pos + i]
            ):
                valid_length += 1
            else:
                break
        return valid_length

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhraseWalker):
            return other.__eq__(self)

        return (
            super().__eq__(other)
            and self.state_machine.phrase == other.state_machine.phrase
        )
