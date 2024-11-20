from __future__ import annotations
from typing import Iterable, Optional, Set, Type

from lexpy import DAWG

from pse.util.state_machine.accepted_state import AcceptedState
from pse.core.walker import Walker
from pse.core.state_machine import StateMachine, StateMachineWalker


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
    ) -> None:
        """
        Initialize the CharAcceptor with a set of valid characters.

        Args:
            charset (Iterable[str]): An iterable of characters to be accepted.
        """
        super().__init__(is_optional=is_optional, is_case_sensitive=case_sensitive)
        self.char_min = char_min or 0
        self.char_limit = char_limit or 0
        self.charset: Set[str] = (
            set(charset)
            if case_sensitive
            else set(char.lower() for char in charset)
        )

    @property
    def walker_class(self) -> Type[Walker]:
        return CharacterWalker

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
        self._raw_value = value

    def get_valid_continuations(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        valid_tokens = set()
        for char in self.acceptor.charset:
            valid_tokens.update(dawg.search_with_prefix(char))

        return valid_tokens

    def should_start_transition(self, token: str) -> bool:
        """Determines if a transition should start with the given input string."""
        token = token.lower() if not self.acceptor.is_case_sensitive else token
        self._accepts_more_input = bool(token and token[0] in self.acceptor.charset)
        return self._accepts_more_input

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

        token = token.lower() if not self.acceptor.is_case_sensitive else token

        # Find valid characters up to char_limit
        valid_length = 0
        for char in token:
            if char not in self.acceptor.charset:
                break
            if self.acceptor.char_limit > 0 and valid_length + self.consumed_character_count >= self.acceptor.char_limit:
                break
            valid_length += 1

        if valid_length == 0:
            self._accepts_more_input = False
            return

        # Create new walker with accumulated value
        new_walker = self.__class__(self.acceptor, f"{self.raw_value}{token[:valid_length]}")
        new_walker.consumed_character_count = self.consumed_character_count + valid_length
        new_walker.remaining_input = token[valid_length:] if valid_length < len(token) else None
        new_walker._accepts_more_input = not new_walker.remaining_input and (
            self.acceptor.char_limit <= 0 or valid_length < self.acceptor.char_limit
        )

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

    # def is_within_value(self) -> bool:
    #     """
    #     Check if the walker has a value.

    #     Returns:
    #         bool: True if the walker has a value, False otherwise.
    #     """
    #     return self.consumed_character_count > 0
