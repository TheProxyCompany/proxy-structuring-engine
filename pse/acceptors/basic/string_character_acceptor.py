from __future__ import annotations
from typing import Iterable, Optional, Set, Type

from pse.core.state_machine import StateMachine
from pse.util.state_machine.accepted_state import AcceptedState
from pse.core.walker import Walker

# INVALID_CHARS is a set containing characters that are not allowed in JSON strings.
# It includes control characters (ASCII 0-31) and the double quote (") and backslash (\) characters.
# These characters need to be escaped or are not permitted in JSON string values.
INVALID_CHARS: Set[str] = {chr(c) for c in range(0, 0x20)} | {'"', "\\"}


class StringCharacterAcceptor(StateMachine):
    """
    Accepts one or more valid JSON unescaped string characters.
    """

    @property
    def walker_class(self) -> Type[Walker]:
        return StringCharacterWalker


class StringCharacterWalker(Walker):
    """
    Walker for navigating through characters in StringCharAcceptor.
    """

    def __init__(
        self,
        acceptor: StringCharacterAcceptor,
        value: Optional[str] = None,
    ) -> None:
        """
        Initialize the Walker.

        Args:
            acceptor (StringCharAcceptor): The parent StringCharAcceptor.
            value (Optional[str]): The accumulated string value. Defaults to None.
        """
        super().__init__(acceptor)
        self.acceptor: StringCharacterAcceptor = acceptor
        self._accepts_more_input = True
        self._raw_value = value

    def can_accept_more_input(self) -> bool:
        return self._accepts_more_input

    def accepts_any_token(self) -> bool:
        return True

    def should_start_transition(self, token: str) -> bool:
        if not token or token[0] in INVALID_CHARS:
            self._accepts_more_input = False
            return False

        return True

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advance the walker with the given input.

        Args:
            token (str): The input to advance with.

        Returns:
            List[Walker]: List of new walkers after advancement.
        """
        # Split input at first invalid character
        valid_prefix = ""
        for char in token:
            if char in INVALID_CHARS:
                break
            valid_prefix += char

        if not valid_prefix:
            self._accepts_more_input = False
            return

        new_walker = self.clone()
        new_walker._raw_value = (self._raw_value or "") + valid_prefix
        new_walker.remaining_input = token[len(valid_prefix):] or None
        new_walker.consumed_character_count += len(valid_prefix)
        yield AcceptedState(new_walker)

    def is_within_value(self) -> bool:
        """
        Check if the walker has a value.

        Returns:
            bool: True if the walker has a value, False otherwise.
        """
        return self._raw_value is not None
