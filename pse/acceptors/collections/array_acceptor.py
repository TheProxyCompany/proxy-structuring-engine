from __future__ import annotations
from typing import List, Any, Optional
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
    StateType,
)
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.json_acceptor import JsonAcceptor


class ArrayAcceptor(StateMachine):
    """
    Accepts a well-formed JSON array and handles state transitions during parsing.

    This acceptor manages the parsing of JSON arrays by defining the state transitions
    and maintaining the current array values being parsed.
    """

    def __init__(
        self,
        graph: Optional[StateMachineGraph] = None,
    ) -> None:
        """
        Initialize the ArrayAcceptor with a state transition graph.

        Args:
            graph (Optional[Dict[StateMachineAcceptor.StateType, List[Tuple[TokenAcceptor, StateMachineAcceptor.StateType]]]], optional):
                Custom state transition graph. If None, a default graph is used to parse JSON arrays.
        """
        if graph is None:
            graph = {
                0: [(TextAcceptor("["), 1)],
                1: [
                    (WhitespaceAcceptor(), 2),
                    (TextAcceptor("]"), "$"),  # Allow empty array
                ],
                2: [(JsonAcceptor({}), 3)],
                3: [(WhitespaceAcceptor(), 4)],
                4: [
                    (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                    (TextAcceptor("]"), "$"),
                ],
            }
        super().__init__(graph)

    def expects_more_input(self, cursor: Cursor) -> bool:
        return cursor.current_state not in self.end_states

    class Cursor(StateMachine.Cursor):
        """
        Cursor for ArrayAcceptor that maintains the current state and accumulated values.
        """

        def __init__(self, acceptor: ArrayAcceptor):
            """
            Initialize the ArrayAcceptor.Cursor with the parent acceptor and an empty list.

            Args:
                acceptor (ArrayAcceptor): The parent ArrayAcceptor instance.
            """
            super().__init__(acceptor)
            self.value: List[Any] = []

        def clone(self) -> ArrayAcceptor.Cursor:
            """
            Clone the current cursor, duplicating its state and accumulated values.

            Returns:
                ArrayAcceptor.Cursor: A new instance of the cloned cursor.
            """
            cloned_cursor = super().clone()
            cloned_cursor.value = self.value[:]
            return cloned_cursor

        def complete_transition(
            self,
            transition_value: Any,
            target_state: StateType,
            is_end_state: bool,
        ) -> bool:
            """
            Handle the completion of a transition by updating the accumulated values.

            Args:
                transition_value (Any): The value transitioned with.
                target_state (StateMachineAcceptor.StateType): The target state after the transition.
                is_end_state (bool): Indicates if the transition leads to an end state.

            Returns:
                bool: True if the transition was successful, False otherwise.
            """
            if target_state == 3 and transition_value is not None:
                self.value.append(transition_value)
            return True

        def get_value(self) -> Any:
            """
            Retrieve the accumulated value from the cursor's history.

            Returns:
                Any: The concatenated values from the accept history and current transition.
            """
            return self.value
