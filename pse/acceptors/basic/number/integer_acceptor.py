from __future__ import annotations

from typing import Any, Iterable, Optional, Union
from pse.acceptors.basic.character_acceptors import DigitAcceptor, CharacterWalker


class IntegerAcceptor(DigitAcceptor):
    """
    Accepts an integer as per JSON specification.
    """

    def __repr__(self) -> str:
        return "IntegerAcceptor()"

    def get_walkers(self) -> Iterable[IntegerWalker]:
        yield IntegerWalker(self)


class IntegerWalker(CharacterWalker):
    """
    Walker for IntegerAcceptor.
    """

    def __init__(self, acceptor: IntegerAcceptor, value: Optional[str] = None) -> None:
        super().__init__(acceptor, value)
        self.acceptor = acceptor
        self.text: str = value or ""
        self.value: Optional[int] = None

    def can_accept_more_input(self) -> bool:
        return self._accepts_remaining_input

    def should_start_transition(self, token: str) -> bool:
        if not token:
            return False

        if not token[0].isdigit():
            self._accepts_remaining_input = False
            return False

        return True

    def should_complete_transition(
        self, transition_value: str, target_state: Any, is_end_state: bool
    ) -> bool:
        """
        Complete the transition to the next state.

        Args:
            transition_value: The value obtained from the transition.
            target_state: The target state after transition.
            is_end_state: Whether the target state is an accepting state.

        Returns:
            bool: True to indicate the transition is complete.
        """
        if not transition_value[0].isdigit():
            self._accepts_remaining_input = False
            return False

        self.text += transition_value
        self.current_state = target_state

        try:
            self.value = int(self.text)
        except ValueError:
            self.value = None

        if self._accepts_remaining_input and not is_end_state:
            return False

        return self.value is not None

    def accumulated_value(self) -> Union[str, int]:
        """
        Retrieve the accumulated integer value.

        Returns:
            Union[str, int]: The integer value if parsed successfully; otherwise, the text.
        """
        try:
            self.value = int(self.text)
            return self.value
        except ValueError:
            return self.text
