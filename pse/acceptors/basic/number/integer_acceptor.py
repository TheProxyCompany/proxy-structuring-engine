from __future__ import annotations

from typing import Any, Iterable, Optional
from pse.acceptors.basic.character_acceptor import CharacterAcceptor, CharacterWalker


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

    def should_complete_transition(self) -> bool:
        """
        Complete the transition to the next state.

        Args:
            transition_value: The value obtained from the transition.
            target_state: The target state after transition.
            is_end_state: Whether the target state is an accepting state.

        Returns:
            bool: True to indicate the transition is complete.
        """
        if not self.transition_walker or not self.transition_walker.raw_value:
            return False

        if not self.transition_walker.raw_value[0].isdigit():
            self._accepts_more_input = False
            return False

        self._raw_value = (self._raw_value or "") + self.transition_walker.raw_value

        if (
            self._accepts_more_input
            and self.target_state
            and self.target_state not in self.acceptor.end_states
        ):
            return False

        return True

    @property
    def current_value(self) -> str | None:
        return self._parse_value(self._raw_value)

    def _parse_value(self, value: Any) -> Any:
        if self.acceptor.drop_leading_zeros:
            return super()._parse_value(value)
        return self._raw_value
