from __future__ import annotations

from typing import Any

from pse_core import State, StateGraph
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.basic.text_acceptor import TextAcceptor
from pse.state_machines.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.state_machines.collections.sequence_acceptor import SequenceAcceptor
from pse.state_machines.json.json_acceptor import JsonAcceptor


class ArrayAcceptor(StateMachine):
    """
    Accepts a well-formed JSON array and handles state transitions during parsing.

    This state_machine manages the parsing of JSON arrays by defining the state transitions
    and maintaining the current array values being parsed.
    """

    def __init__(self, state_graph: StateGraph | None = None) -> None:
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
            ],
        }
        super().__init__(state_graph or base_array_state_graph)

    def get_new_walker(self, state: State | None = None) -> ArrayWalker:
        return ArrayWalker(self, state)


class ArrayWalker(Walker):
    """
    Walker for ArrayAcceptor that maintains the current state and accumulated values.
    """

    def __init__(
        self, state_machine: ArrayAcceptor, current_state: State | None = None
    ):
        """
        Initialize the ArrayAcceptor.Walker with the parent state_machine and an empty list.

        Args:
            state_machine (ArrayAcceptor): The parent ArrayAcceptor instance.
        """
        super().__init__(state_machine, current_state)
        self.state_machine: ArrayAcceptor = state_machine
        self.value: list[Any] = []

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
            and self.transition_walker.get_raw_value() is not None
        ):
            self.value.append(self.transition_walker.get_raw_value())

        return super().should_complete_transition()

    def parse_value(self, value: str | None) -> Any:
        import json

        if not value:
            return None
        return json.loads(value)
