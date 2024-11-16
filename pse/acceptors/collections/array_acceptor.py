from __future__ import annotations
from typing import List, Any, Type, Optional
from pse.util.state_machine.types import StateGraph
from pse.core.state_machine import StateMachine, StateMachineWalker
from pse.core.walker import Walker
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

    def __init__(self, state_graph: Optional[StateGraph] = None) -> None:
        """
        Initialize the ArrayAcceptor with a state transition graph.

        Args:
            graph (Optional[Dict[StateMachineAcceptor.StateType, List[Tuple[TokenAcceptor, StateMachineAcceptor.StateType]]]], optional):
                Custom state transition graph. If None, a default graph is used to parse JSON arrays.
        """
        base_array_state_graph: StateGraph = {
            0: [(TextAcceptor("["), 1)],
            1: [
                (WhitespaceAcceptor(), 2),
                (TextAcceptor("]"), "$"),  # Allow empty array
            ],
            2: [(JsonAcceptor(), 3)],
            3: [(WhitespaceAcceptor(), 4)],
            4: [
                (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                (TextAcceptor("]"), "$"),
            ]
        }
        super().__init__(state_graph or base_array_state_graph)

    @property
    def walker_class(self) -> Type[Walker]:
        return ArrayWalker


class ArrayWalker(StateMachineWalker):
    """
    Walker for ArrayAcceptor that maintains the current state and accumulated values.
    """

    def __init__(self, acceptor: ArrayAcceptor, current_state: int = 0):
        """
        Initialize the ArrayAcceptor.Walker with the parent acceptor and an empty list.

        Args:
            acceptor (ArrayAcceptor): The parent ArrayAcceptor instance.
        """
        super().__init__(acceptor, current_state)
        self.value: List[Any] = []

    def clone(self) -> ArrayWalker:
        """
        Clone the current walker, duplicating its state and accumulated values.

        Returns:
            ArrayAcceptor.Walker: A new instance of the cloned walker.
        """
        cloned_walker = super().clone()
        cloned_walker.value = self.value[:]
        return cloned_walker

    def should_complete_transition(self) -> bool:
        """
        Handle the completion of a transition by updating the accumulated values.

        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if (
            self.target_state == 3
            and self.transition_walker
            and self.transition_walker.raw_value is not None
        ):
            self.value.append(self.transition_walker.raw_value)

        return super().should_complete_transition()
