from __future__ import annotations

import logging

from pse_core.accepted_state import AcceptedState
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

logger = logging.getLogger(__name__)


class TextAcceptor(StateMachine):
    """
    Accepts a predefined sequence of characters, validating input against the specified text.

    Attributes:
        text (str): The target string that this state_machine is validating against.
    """

    def __init__(
        self, text: str, is_optional: bool = False, is_case_sensitive: bool = True
    ):
        """
        Initialize a new TextAcceptor instance with the specified text.

        Args:
            text (str): The string of characters that this state_machine will validate.
                Must be a non-empty string.

        Raises:
            ValueError: If the provided text is empty.
        """
        super().__init__(is_optional=is_optional, is_case_sensitive=is_case_sensitive)

        if not text:
            raise ValueError("Text must be a non-empty string.")

        self.text = text

    def get_new_walker(self, state: int | str | None = None) -> TextWalker:
        return TextWalker(self)

    def __repr__(self) -> str:
        """
        Provide a string representation of the TextAcceptor.

        Returns:
            str: A string representation of the TextAcceptor.
        """
        return f"TextAcceptor(text={self.text!r})"


class TextWalker(Walker):
    """
    Represents the current position within the TextAcceptor's text during validation.

    Attributes:
        state_machine (TextAcceptor): The TextAcceptor instance that owns this Walker.
        consumed_character_count (int): The current position in the text being validated.
    """

    def __init__(
        self,
        state_machine: TextAcceptor,
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

        self.target_state = "$"
        self.state_machine: TextAcceptor = state_machine
        self.consumed_character_count = consumed_character_count or 0

    def can_accept_more_input(self) -> bool:
        """
        Check if the walker can accept more input.
        """
        return self._accepts_more_input and self.consumed_character_count < len(self.state_machine.text)

    def should_start_transition(self, token: str) -> bool:
        """
        Start a transition if the token is not empty and matches the remaining text.
        """
        if not token:
            return False

        valid_length = self._get_valid_match_length(token)
        return valid_length > 0

    def get_valid_continuations(self, depth: int = 0) -> list[str]:
        if self.consumed_character_count >= len(self.state_machine.text):
            return []

        valid_continuations = []

        remaining_text = self.state_machine.text[self.consumed_character_count :]
        valid_continuations.append(remaining_text)

        # Check if the exact partial prefixes exist in the DAWG
        max_possible_match_len = len(remaining_text)
        for i in range(1, max_possible_match_len):
            partial = remaining_text[:i]
            valid_continuations.append(partial)

        return valid_continuations

    def consume_token(self, token: str) -> list[Walker]:
        """
        Advances the walker if the token matches the expected text at the current position.
        Args:
            token (str): The string to match against the expected text.

        Returns:
            list[Walker]: A walker if the token matches, empty otherwise.
        """
        valid_length = self._get_valid_match_length(token)

        if not token:
            return []

        if valid_length > 0:
            consumed_length = self.consumed_character_count + valid_length
            remaining_input = token[valid_length:] or None
            next_walker = self.__class__(self.state_machine, consumed_length)
            next_walker.remaining_input = remaining_input
            next_walker._accepts_more_input = consumed_length < len(
                self.state_machine.text
            )

            if consumed_length == len(self.state_machine.text):
                return [AcceptedState(next_walker)]
            else:
                return [next_walker]

        return []

    def get_current_value(self) -> str:
        """
        Retrieves the current state of the text being accepted, highlighting the remaining portion.

        Returns:
            str: The accepted portion of the text
        """
        return self.get_raw_value()

    def get_raw_value(self) -> str:
        return (
            self.state_machine.text[: self.consumed_character_count]
            if self.consumed_character_count < len(self.state_machine.text)
            else self.state_machine.text
        )

    def is_within_value(self) -> bool:
        """
        Determine if the walker is currently within a value.

        Returns:
            bool: True if in a value, False otherwise.
        """
        return (
            self.consumed_character_count > 0
            and self.consumed_character_count < len(self.state_machine.text)
        )

    def _format_current_edge(self) -> str:
        target_state = (
            f"--> ({'✅' if self.target_state == "$" else self.target_state})"
            if self.target_state is not None
            else ""
        )
        seen = self.state_machine.text[: self.consumed_character_count]
        remaining = self.state_machine.text[self.consumed_character_count :]
        accumulated_value = seen + (f"👉{remaining}" if remaining else "")
        return (
            f"Current edge: ({self.current_state}) --"
            + repr(accumulated_value)[1:-1]
            + target_state
        )

    def _get_valid_match_length(self, token: str, pos: int | None = None) -> int:
        pos = pos or self.consumed_character_count
        valid_length = 0
        for i, c in enumerate(token):
            if pos + i < len(self.state_machine.text) and c == self.state_machine.text[pos + i]:
                valid_length += 1
            else:
                break
        return valid_length

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextWalker):
            return other.__eq__(self)

        return (
            super().__eq__(other)
            and self.state_machine.text == other.state_machine.text
        )
