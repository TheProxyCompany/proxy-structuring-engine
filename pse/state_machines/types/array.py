from __future__ import annotations

from typing import Any

from pse_core import State, StateGraph
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.json import JsonStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class ArrayStateMachine(StateMachine):
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
            0: [(PhraseStateMachine("["), 1)],
            1: [
                (WhitespaceStateMachine(), 2),
                (PhraseStateMachine("]"), "$"),  # Allow empty array
            ],
            2: [(JsonStateMachine(), 3)],
            3: [(WhitespaceStateMachine(), 4)],
            4: [
                (
                    ChainStateMachine(
                        [
                            PhraseStateMachine(","),
                            WhitespaceStateMachine()
                        ]
                    ), 2,
                ),
                (PhraseStateMachine("]"), "$"),
            ],
        }
        super().__init__(state_graph or base_array_state_graph)

    def get_new_walker(self, state: State | None = None) -> ArrayWalker:
        return ArrayWalker(self, state)

    def __str__(self) -> str:
        return "Array"


class ArrayWalker(Walker):
    """
    Walker for ArrayAcceptor that maintains the current state and accumulated values.
    """

    def __init__(
        self,
        state_machine: ArrayStateMachine,
        current_state: State | None = None,
    ):
        """
        Initialize the ArrayAcceptor.Walker with the parent state_machine and an empty list.

        Args:
            state_machine (ArrayAcceptor): The parent ArrayAcceptor instance.
        """
        super().__init__(state_machine, current_state)
        self.state_machine: ArrayStateMachine = state_machine
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

    def is_within_value(self) -> bool:
        return self.current_state == 3

    def add_to_history(self, walker: Walker) -> None:
        if self.is_within_value():
            self.value.append(walker.get_current_value())
        super().add_to_history(walker)

    def get_current_value(self) -> list:
        """
        Get the current parsed JSON object.

        Returns:
            dict[str, Any]: The accumulated key-value pairs representing the JSON object.
        """
        if not self.get_raw_value():
            return []
        return self.value
