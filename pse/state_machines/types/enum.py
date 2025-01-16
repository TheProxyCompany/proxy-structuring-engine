from __future__ import annotations

import json

from pse_core import State, StateGraph
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine


class EnumStateMachine(StateMachine):
    """
    Accept one of several constant strings.
    """

    def __init__(self, enum_values: list[str], require_quotes: bool = True) -> None:
        """
        Initialize the EnumSchemaAcceptor with a dictionary-based transition graph.

        Args:
            schema (dict): A dictionary containing the 'enum' key with a list of allowed values.
            require_quotes (bool):
                Flag to determine if enum values should be wrapped in quotes.
                Defaults to True.

        Raises:
            KeyError: If the 'enum' key is not present in the schema.
            TypeError: If the 'enum' value is not a list.
        """
        if not enum_values:
            raise ValueError("Enum values must be provided.")

        state_graph: StateGraph = {0: []}
        for value in enum_values:
            enum_value = json.dumps(value) if require_quotes else value
            state_graph[0].append((PhraseStateMachine(enum_value), "$"))

        super().__init__(state_graph)

    def get_walkers(self, state: State | None = None) -> list[Walker]:
        walkers = []
        for edge, _ in self.get_edges(state or 0):
            walkers.extend(edge.get_walkers())
        return walkers
