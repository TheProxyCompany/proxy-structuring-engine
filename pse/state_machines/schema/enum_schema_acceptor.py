from __future__ import annotations

import json

from pse_core.state_machine import StateMachine

from pse.state_machines.basic.text_acceptor import TextAcceptor


class EnumSchemaAcceptor(StateMachine):
    """
    Accept one of several constant strings.
    """

    def __init__(self, schema: dict) -> None:
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
        enum_values: list[str] | None = schema.get("enum")

        if enum_values is None:
            raise KeyError("Schema must contain 'enum' key.")

        if not isinstance(enum_values, list):
            raise TypeError("'enum' must be a list of string values.")

        super().__init__(
            {
                0: [(TextAcceptor(json.dumps(value)), "$") for value in enum_values],
            },
        )
