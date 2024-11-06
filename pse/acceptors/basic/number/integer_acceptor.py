from __future__ import annotations

from typing import Any, Iterable, Optional
from pse.acceptors.basic.character_acceptors import CharacterAcceptor,  CharacterWalker

class IntegerAcceptor(CharacterAcceptor):
    """
    Accepts an integer as per JSON specification.
    """

    def __init__(self, drop_leading_zeros: bool = True) -> None:
        super().__init__("0123456789")
        self.drop_leading_zeros = drop_leading_zeros

    def get_walkers(self) -> Iterable[IntegerWalker]:
        yield IntegerWalker(self)


class IntegerWalker(CharacterWalker):
    """
    Walker for IntegerAcceptor.
    """

    def __init__(self, acceptor: IntegerAcceptor, value: Optional[str] = None) -> None:
        super().__init__(acceptor, value)
        self.acceptor: IntegerAcceptor = acceptor
        self._raw_value = value or ""

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
        self, transition_value: str, is_end_state: bool
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

        self._raw_value = (self._raw_value or "") + transition_value

        if self._accepts_remaining_input and not is_end_state:
            return False

        return True

    def _parse_value(self, value: Any) -> Any:
        if self.acceptor.drop_leading_zeros:
            return super()._parse_value(value)
        return self._raw_value
