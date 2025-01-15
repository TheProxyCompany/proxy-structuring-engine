from __future__ import annotations

from pse_core import StateGraph
from pse_core.state_machine import StateMachine

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
            enum_value = repr(value) if require_quotes else value
            state_graph[0].append((PhraseStateMachine(enum_value), "$"))

        super().__init__(state_graph)

    def __str__(self) -> str:
        return "Enum"
