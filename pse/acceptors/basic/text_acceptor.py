from __future__ import annotations

from lexpy import DAWG

from pse.state_machine.walker import Walker
from pse.state_machine.state_machine import StateMachine, StateMachineWalker
from pse.state_machine.accepted_state import AcceptedState
from typing import Iterable, Set
import logging

logger = logging.getLogger(__name__)


class TextAcceptor(StateMachine):
    """
    Accepts a predefined sequence of characters, validating input against the specified text.

    Attributes:
        text (str): The target string that this acceptor is validating against.
    """

    def __init__(
        self, text: str, is_optional: bool = False, is_case_sensitive: bool = True
    ):
        """
        Initialize a new TextAcceptor instance with the specified text.

        Args:
            text (str): The string of characters that this acceptor will validate.
                Must be a non-empty string.

        Raises:
            ValueError: If the provided text is empty.
        """
        super().__init__(
            walker_type=TextWalker,
            is_optional=is_optional,
            is_case_sensitive=is_case_sensitive,
        )

        self.text = text

    def get_walkers(self) -> Iterable[TextWalker]:
        """
        Get one or more walkers to traverse the acceptor.
        Override.
        """
        yield TextWalker(self)

    def __repr__(self) -> str:
        """
        Provide a string representation of the TextAcceptor.

        Returns:
            str: A string representation of the TextAcceptor.
        """
        return f"TextAcceptor(matching={repr(self.text)})"


class TextWalker(StateMachineWalker):
    """
    Represents the current position within the TextAcceptor's text during validation.

    Attributes:
        acceptor (TextAcceptor): The TextAcceptor instance that owns this Walker.
        consumed_character_count (int): The current position in the text being validated.
    """

    def __init__(self, acceptor: TextAcceptor, consumed_character_count: int = 0):
        """
        Initialize a new Walker instance.

        Args:
            acceptor (TextAcceptor): The TextAcceptor instance associated with this walker.
            consumed_character_count (int, optional): The initial position in the text. Defaults to 0.
        """
        super().__init__(acceptor)
        self.acceptor = acceptor
        self.consumed_character_count = consumed_character_count

    def can_accept_more_input(self) -> bool:
        """
        Check if the walker can accept more input.
        """
        return self.consumed_character_count < len(self.acceptor.text)

    def should_start_transition(self, token: str) -> bool:
        """
        Start a transition if the token is not empty and matches the remaining text.

        Example:
        ```
            acceptor.text = "hello"
            token = "hell"
            start_transition(token) -> True
        ```
        """
        if not token:
            return False

        remaining_text = self.acceptor.text[self.consumed_character_count :]
        should_start_transition = remaining_text.startswith(token) or token.startswith(
            remaining_text
        )
        logger.debug(
            f"{self} should {'' if should_start_transition else 'not '}start transition {repr(token)}"
        )

        return should_start_transition

    def should_complete_transition(
        self, transition_value: str, is_end_state: bool
    ) -> bool:
        """
        Complete the transition if the transition value matches the remaining text.
        """
        return transition_value == self.acceptor.text[self.consumed_character_count :]

    def select(self, dawg: DAWG, depth: int = 0) -> Set[str]:
        """
        Selects prefix matches of the target text from the current position.

        Args:
            dawg (DAWG): The DAWG to select from.
            depth (int): The current depth of the walker in the state machine.
        Returns:
            Set[str]: A set containing exact partial matches of the target text.
        """
        if self.consumed_character_count >= len(self.acceptor.text):
            return set()

        remaining_text = self.acceptor.text[self.consumed_character_count :]
        results = set()

        # Only check if the exact partial text exists in the DAWG
        if remaining_text in dawg:
            results.add(remaining_text)

        # Check if the exact partial prefixes exist in the DAWG
        max_possible_match_len = min(len(remaining_text), 8)
        for i in range(1, max_possible_match_len):
            partial = remaining_text[:i]
            if partial in dawg:
                results.add(partial)

        return results

    def consume_token(self, token: str) -> Iterable[Walker]:
        """
        Advances the walker if the given value matches the expected text at the current position.
        Args:
            value (str): The string to match against the expected text.

        Returns:
            Iterable[TextAcceptorwalker]: A list containing the next walker if the value matches,
                                or an empty list otherwise.
        """
        expected_text = self.acceptor.text
        pos = self.consumed_character_count

        max_possible_match_len = len(expected_text) - pos
        input_len = len(token)
        match_len = min(max_possible_match_len, input_len)

        # Get the segment to compare
        expected_segment = expected_text[pos : pos + match_len]
        valid_prefix = token[:match_len]

        if expected_segment == valid_prefix:
            new_pos = pos + match_len
            remaining_input = token[match_len:]

            next_walker = self.__class__(self.acceptor, new_pos)
            if remaining_input:
                logger.debug(f"match found, remaining input: {remaining_input}")
                next_walker.remaining_input = remaining_input

            if new_pos == len(expected_text):
                yield AcceptedState(next_walker)
            else:
                yield next_walker

    def accumulated_value(self) -> str:
        """
        Retrieves the current state of the text being accepted, highlighting the remaining portion.

        Returns:
            str: The accepted portion of the text followed by a marker and the remaining text,
                    e.g., 'hel👉lo' if consumed_character_count is 3.
        """
        return (
            f"{self.acceptor.text[:self.consumed_character_count]}👉{self.acceptor.text[self.consumed_character_count:]}"
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

    def __repr__(self) -> str:
        """
        Provide a string representation of the Walker.

        Returns:
            str: A string representation of the Walker.
        """
        value = (
            f"{self.acceptor.text[:self.consumed_character_count]}👉{self.acceptor.text[self.consumed_character_count:]}"
            if self.consumed_character_count < len(self.acceptor.text)
            else self.acceptor.text
        )
        if self.consumed_character_count == len(self.acceptor.text):
            return f"{self.acceptor}"
        else:
            return f"Text.Walker(value=`{value}`)"
