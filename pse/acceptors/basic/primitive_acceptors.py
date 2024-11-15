from __future__ import annotations
from typing import Iterable, Optional
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
    StateMachineWalker,
)
from pse.acceptors.basic.text_acceptor import TextAcceptor, TextWalker
from lexpy import DAWG


class BooleanAcceptor(StateMachine):
    """
    Accepts a JSON boolean value: true, false.
    """

    def __init__(self) -> None:
        """
        Initialize the BooleanAcceptor with its state transitions defined as a state graph.
        """
        graph: StateMachineGraph = {
            0: [
                (TextAcceptor("true"), "$"),
                (TextAcceptor("false"), "$"),
            ]
        }
        super().__init__(graph, walker_type=BooleanWalker)


class BooleanWalker(StateMachineWalker):
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
            and self.target_state in self.acceptor.end_states
            and self.transition_walker
        ):
            self._raw_value = self.transition_walker.raw_value
        return self._raw_value == "true" or self._raw_value == "false"

    @property
    def current_value(self) -> Optional[bool]:
        """
        Get the parsed boolean value.

        Returns:
            Optional[bool]: The parsed boolean or None if not yet parsed.
        """
        return self._raw_value == "true" if self._raw_value is not None else None


class NullAcceptor(TextAcceptor):
    """
    Accepts the JSON null value.
    """

    def __init__(self) -> None:
        """
        Initialize the NullAcceptor with the text 'null'.
        """
        super().__init__("null")

    def get_walkers(self) -> Iterable[NullWalker]:
        yield NullWalker(self)


class NullWalker(TextWalker):
    """
    Walker for NullAcceptor to track parsing state.
    """

    def select(self, dawg: DAWG) -> Iterable[str]:
        yield "null"

    @property
    def current_value(self) -> str:
        return "null"
