from __future__ import annotations

from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.basic.text_acceptor import TextAcceptor


class BooleanAcceptor(StateMachine):
    """
    Accepts a JSON boolean value: true, false.
    """

    def __init__(self) -> None:
        """
        Initialize the BooleanAcceptor with its state transitions defined as a state graph.
        """
        super().__init__(
            {
                0: [
                    (TextAcceptor("true"), "$"),
                    (TextAcceptor("false"), "$"),
                ]
            }
        )

    def get_new_walker(self, state: int | str) -> BooleanWalker:
        return BooleanWalker(self, state)


class BooleanWalker(Walker):
    """
    Walker for BooleanAcceptor to track parsing state and value.
    """

    def should_complete_transition(self) -> bool:
        """
        Handle the completion of a transition.

        Args:
            transition_value (str): The value transitioned with.
            target_state (Any): The target state after transition.
            is_end_state (bool): Indicates if the transition leads to an end state.

        Returns:
            bool: Success of the transition.
        """
        if (
            self.target_state
            and self.target_state in self.state_machine.end_states
            and self.transition_walker
        ):
            self._raw_value = self.transition_walker.raw_value
        return self._raw_value == "true" or self._raw_value == "false"

    @property
    def current_value(self) -> bool | None:
        """
        Get the parsed boolean value.

        Returns:
            Optional[bool]: The parsed boolean or None if not yet parsed.
        """
        return self._raw_value == "true" if self._raw_value is not None else None
