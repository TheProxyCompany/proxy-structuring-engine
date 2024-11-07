from __future__ import annotations
from typing import Iterable, Optional
from pse.state_machine.walker import Walker
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

    def get_walkers(self) -> Iterable[Walker]:
        yield self._walker(self)


class BooleanWalker(StateMachineWalker):
    """
    Walker for BooleanAcceptor to track parsing state and value.
    """

    def __init__(self, acceptor: BooleanAcceptor) -> None:
        super().__init__(acceptor)

    def should_complete_transition(
        self,
        transition_value: str,
        is_end_state: bool,
    ) -> bool:
        """
        Handle the completion of a transition.

        Args:
            transition_value (str): The value transitioned with.
            target_state (Any): The target state after transition.
            is_end_state (bool): Indicates if the transition leads to an end state.

        Returns:
            bool: Success of the transition.
        """
        if is_end_state:
            # Assign True if transition_value is "true", else False
            self._raw_value = transition_value
        return True

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

    def __repr__(self) -> str:
        return "NullAcceptor()"

    def get_walkers(self) -> Iterable[NullWalker]:
        yield NullWalker(self)


class NullWalker(TextWalker):
    """
    Walker for NullAcceptor to track parsing state.
    """

    def select(self, dawg: DAWG) -> Iterable[str]:
        yield "null"

    def current_value(self) -> str:
        return "null"
