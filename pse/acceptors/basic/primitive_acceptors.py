from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pse.state_machine.cursor import Cursor
from pse.state_machine.state_machine import (
    StateMachine,
    StateType,
)
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.token_acceptor import TokenAcceptor
from lexpy import DAWG

class BooleanAcceptor(StateMachine):
    """
    Accepts a JSON boolean value: true, false.
    """

    def __init__(self) -> None:
        """
        Initialize the BooleanAcceptor with its state transitions defined as a state graph.
        """
        # Define the state graph where state 0 transitions to end state "$" on "true" or "false"
        graph: Dict[StateType, List[Tuple[TokenAcceptor, StateType]]] = {
            0: [(TextAcceptor("true"), "$"), (TextAcceptor("false"), "$")]
        }
        # Initialize the StateMachineAcceptor with the defined graph, initial state, and end states
        super().__init__(graph=graph, initial_state=0, end_states={"$"})

    def expects_more_input(self, cursor: Cursor) -> bool:
        return False

    class Cursor(StateMachine.Cursor):
        """
        Cursor for BooleanAcceptor to track parsing state and value.
        """

        def __init__(self, acceptor: BooleanAcceptor) -> None:
            """
            Initialize the cursor.

            Args:
                acceptor (BooleanAcceptor): The parent acceptor.
            """
            super().__init__(acceptor)
            self.value: Optional[bool] = None

        def complete_transition(
            self,
            transition_value: str,
            target_state: Any,
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
                self.value = transition_value == "true"
            return True

        def get_value(self) -> Optional[bool]:
            """
            Get the parsed boolean value.

            Returns:
                Optional[bool]: The parsed boolean or None if not yet parsed.
            """
            return self.value


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

    def expects_more_input(self, cursor: Cursor) -> bool:
        return False

    class Cursor(TextAcceptor.Cursor):
        """
        Cursor for NullAcceptor to track parsing state.
        """

        def select(self, dawg: DAWG) -> Iterable[str]:
            yield "null"

        def get_value(self) -> str:
            return "null"
