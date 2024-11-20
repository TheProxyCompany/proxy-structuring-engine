from __future__ import annotations

from lexpy import DAWG

from pse.core.walker import Walker
from pse.core.state_machine import StateMachine, StateMachineWalker
from pse.util.state_machine.accepted_state import AcceptedState
from typing import Iterable, Optional, Set, Type
import logging

logger = logging.getLogger(__name__)


class TextAcceptor(StateMachine):
    """
    Accepts a predefined sequence of characters, validating input against the specified text.

    Attributes:
        text (str): The target string that this acceptor is validating against.
    """

    def __init__(
        self,
        text: str,
        is_optional: bool = False,
        is_case_sensitive: bool = True
    ):
        """
        Initialize a new TextAcceptor instance with the specified text.

        Args:
            text (str): The string of characters that this acceptor will validate.
                Must be a non-empty string.

        Raises:
            ValueError: If the provided text is empty.
        """
        super().__init__(is_optional=is_optional, is_case_sensitive=is_case_sensitive)

        if not text:
            raise ValueError("Text must be a non-empty string.")

        self.text = text

    @property
    def walker_class(self) -> Type[Walker]:
        return TextWalker

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        """
        Provide a string representation of the TextAcceptor.

        Returns:
            str: A string representation of the TextAcceptor.
        """
        return f"TextAcceptor({repr(self.text)})"


class TextWalker(StateMachineWalker):
    """
    Represents the current position within the TextAcceptor's text during validation.

    Attributes:
        acceptor (TextAcceptor): The TextAcceptor instance that owns this Walker.
        consumed_character_count (int): The current position in the text being validated.
    """

    def __init__(
        self,
        acceptor: TextAcceptor,
        consumed_character_count: Optional[int] = None,
    ):
        """
        Initialize a new Walker instance.

        Args:
            acceptor (TextAcceptor): The TextAcceptor instance associated with this walker.
            consumed_character_count (int, optional): The initial position in the text. Defaults to 0.
        """
        super().__init__(acceptor)
        self.acceptor = acceptor
        self.consumed_character_count = consumed_character_count or 0

    def can_accept_more_input(self) -> bool:
        """
        Check if the walker can accept more input.
        """
        return self._accepts_more_input and self.consumed_character_count < len(
            self.acceptor.text
        )

    def should_start_transition(self, token: str) -> bool:
        """
        Start a transition if the token is not empty and matches the remaining text.
        """
        if not token:
            return False

        remaining_text = self.acceptor.text[self.consumed_character_count :]
        return remaining_text.startswith(token) or token.startswith(remaining_text)

    def get_valid_continuations(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        results = set()
        if self.consumed_character_count >= len(self.acceptor.text):
            return results

        remaining_text = self.acceptor.text[self.consumed_character_count :]
        # Only check if the exact partial text exists in the DAWG
        if remaining_text in dawg:
            results.add(remaining_text)

        # Check if the exact partial prefixes exist in the DAWG
        max_possible_match_len = min(len(remaining_text), 8)
        for i in range(1, max_possible_match_len):
            partial = remaining_text[:i]
            if partial in dawg:
                # if partial is a token (exists in DAWG), add it to the results
                results.add(partial)

        return results

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advances the walker if the token matches the expected text at the current position.
        Args:
            token (str): The string to match against the expected text.

        Returns:
            Iterable[Walker]: A walker if the token matches, empty otherwise.
        """
        pos = self.consumed_character_count
        match_len = min(len(self.acceptor.text) - pos, len(token))

        if self.acceptor.text[pos : pos + match_len] == token[:match_len]:
            next_walker = self.__class__(self.acceptor, pos + match_len)
            next_walker.remaining_input = token[match_len:] or None
            next_walker._accepts_more_input = (pos + match_len) < len(self.acceptor.text)

            if pos + match_len == len(self.acceptor.text):
                yield AcceptedState(next_walker)
            else:
                yield next_walker

    @property
    def current_value(self) -> str:
        """
        Retrieves the current state of the text being accepted, highlighting the remaining portion.

        Returns:
            str: The accepted portion of the text
        """
        return self.raw_value

    @property
    def raw_value(self) -> str:
        return (
            self.acceptor.text[: self.consumed_character_count]
            if self.consumed_character_count < len(self.acceptor.text)
            else self.acceptor.text
        )

    def is_within_value(self) -> bool:
        """
        Determine if the walker is currently within a value.

        Returns:
            bool: True if in a value, False otherwise.
        """
        return (
            self.consumed_character_count > 0
            and self.consumed_character_count < len(self.acceptor.text)
        )
