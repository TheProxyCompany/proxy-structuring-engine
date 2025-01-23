from __future__ import annotations

from typing import Any

from pse_core import State
from pse_core.walker import Walker

from pse.state_machines import get_state_machine
from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.array import ArrayStateMachine, ArrayWalker
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class ArraySchemaStateMachine(ArrayStateMachine):
    def __init__(self, schema: dict[str, Any], context: dict[str, Any]) -> None:
        self.schema = schema
        self.context = context
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("["), 1),
                ],
                1: [
                    (WhitespaceStateMachine(), 2),
                    (PhraseStateMachine("]"), "$"),
                ],
                2: [
                    (get_state_machine(self.schema["items"], self.context), 3),
                ],
                3: [
                    (WhitespaceStateMachine(), 4),
                ],
                4: [
                    (
                        ChainStateMachine([
                                PhraseStateMachine(","),
                                WhitespaceStateMachine()
                            ]
                        ),
                        2,
                    ),
                    (PhraseStateMachine("]"), "$"),
                ],
            }
        )

    def get_transitions(self, walker: Walker) -> list[tuple[Walker, State]]:
        """Retrieve transition walkers from the current state.

        For each edge from the current state, returns walkers that can traverse that edge.
        Handles optional acceptors and pass-through transitions appropriately.

        Args:
            walker: The walker initiating the transition.
            state: Optional starting state. If None, uses the walker's current state.

        Returns:
            list[tuple[Walker, State]]: A list of tuples representing transitions.
        """
        if walker.current_state == 4:
            transitions = []
            if len(walker.get_current_value()) >= self.min_items():
                for transition in PhraseStateMachine("]").get_walkers():
                    transitions.append((transition, "$"))

            if len(walker.get_current_value()) < self.max_items():
                for transition in ChainStateMachine([PhraseStateMachine(","), WhitespaceStateMachine()]).get_walkers():
                    transitions.append((transition, 2))

            return transitions
        elif walker.current_state == 1 and self.min_items() > 0:
            transitions = []
            for transition in WhitespaceStateMachine().get_walkers():
                transitions.append((transition, 2))
            return transitions
        else:
            return super().get_transitions(walker)

    def get_new_walker(self, state: State | None = None) -> ArraySchemaWalker:
        return ArraySchemaWalker(self, state)

    def min_items(self) -> int:
        """
        Returns the minimum number of items in the array, according to the schema
        """
        return self.schema.get("minItems", 0)

    def max_items(self) -> int:
        """
        Returns the maximum number of items in the array, according to the schema
        """
        return self.schema.get("maxItems", 2**32)

    def __str__(self) -> str:
        return super().__str__() + "Schema"

class ArraySchemaWalker(ArrayWalker):
    """
    Walker for ArrayAcceptor
    """

    def __init__(
        self,
        state_machine: ArraySchemaStateMachine,
        current_state: State | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: ArraySchemaStateMachine = state_machine
