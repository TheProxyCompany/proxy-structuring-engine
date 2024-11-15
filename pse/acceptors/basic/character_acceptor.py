from __future__ import annotations
from typing import Iterable, Optional, Set, Type

from lexpy import DAWG

from pse.state_machine.accepted_state import AcceptedState
from pse.state_machine.walker import Walker
from pse.state_machine.state_machine import StateMachine, StateMachineWalker


class CharacterAcceptor(StateMachine):
    """
    Accept multiple characters at once if they are all in the charset.
    Will also prefix the walker with the valid characters if it's not in the
    accepted state.
    """

    def __init__(
        self,
        charset: Iterable[str],
        char_min: Optional[int] = None,
        char_limit: Optional[int] = None,
        is_optional: bool = False,
        case_sensitive: bool = True,
        walker_type: Optional[Type[CharacterWalker]] = None,
    ) -> None:
        """
        Initialize the CharAcceptor with a set of valid characters.

        Args:
            charset (Iterable[str]): An iterable of characters to be accepted.
        """
        super().__init__(
            graph={},
            walker_type=walker_type or CharacterWalker,
            is_optional=is_optional,
            is_case_sensitive=case_sensitive,
        )
        self.charset: Set[str] = (
            set(charset) if case_sensitive else set(char.lower() for char in charset)
        )
        self.char_min = char_min or 0
        self.char_limit = char_limit or 0

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        sorted_character_set = ", ".join(sorted(self.charset))
        return f"{self.__class__.__name__}(charset=[{sorted_character_set}])"


class CharacterWalker(StateMachineWalker):
    """
    Walker for navigating through characters in CharAcceptor.
    """

    def __init__(
        self,
        acceptor: CharacterAcceptor,
        value: Optional[str] = None,
    ) -> None:
        """
        Initialize the Walker.

        Args:
            acceptor (CharAcceptor): The parent CharAcceptor.
            value (Optional[str]): The current input value. Defaults to None.
        """
        super().__init__(acceptor)
        self.acceptor: CharacterAcceptor = acceptor
        self._raw_value = value or ""

    def can_accept_more_input(self) -> bool:
        return self._accepts_more_input

    def select(self, dawg: DAWG) -> Iterable[str]:
        """
        Select characters that are valid based on the acceptor's charset.

        Args:
            candidate_chars (Set[str]): Set of candidate characters (ignored in this implementation).

        Returns:
            Iterable[str]: An iterable of valid characters from the acceptor's charset.
        """
        for char in self.acceptor.charset:
            yield char

    def should_start_transition(self, token: str) -> bool:
        """Determines if a transition should start with the given input string.

        Args:
            token: The input string to process.

        Returns:
            True if the transition should start; False otherwise.
        """
        if not self.acceptor.is_case_sensitive:
            token = token.lower()

        if not token or token[0] not in self.acceptor.charset:
            self._accepts_more_input = False
            return False

        return True

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advance the walker with the given input. Accumulates all valid characters.

        Args:
            token (str): The input to advance with.

        Returns:
            Iterable[Walker]: An iterable containing the new walker state if input is valid.
        """

        if not token:
            self._accepts_more_input = False
            return

        if not self.acceptor.is_case_sensitive:
            token = token.lower()

        valid_length = 0
        accumulated_length = self.consumed_character_count
        for character in token:
            if character not in self.acceptor.charset:
                break
            if (
                self.acceptor.char_limit > 0
                and accumulated_length >= self.acceptor.char_limit
            ):
                break
            accumulated_length += 1
            valid_length += 1

        valid_characters = token[:valid_length]
        remaining_input = token[valid_length:] if valid_length < len(token) else None

        if not valid_characters:
            self._accepts_more_input = False
            return

        # Accumulate valid characters with existing value
        accumulated_value = f"{self._raw_value or ''}{valid_characters}"

        new_walker = self.__class__(self.acceptor, accumulated_value)
        new_walker.consumed_character_count = accumulated_length
        new_walker.remaining_input = remaining_input

        if self.acceptor.char_limit > 0:
            new_walker._accepts_more_input = (
                not remaining_input and valid_length < self.acceptor.char_limit
            )
        else:
            new_walker._accepts_more_input = not remaining_input

        yield (
            AcceptedState(new_walker)
            if new_walker.consumed_character_count >= self.acceptor.char_min
            else new_walker
        )

    @property
    def raw_value(self) -> str:
        return self._raw_value or ""

    @property
    def current_value(self) -> Optional[str]:
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
