from __future__ import annotations

from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.core.state_machine import StateMachine

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
        enum_values = schema.get("enum")

        if enum_values is None:
            raise KeyError("Schema must contain 'enum' key.")

        if not isinstance(enum_values, list):
            raise TypeError("'enum' must be a list of string values.")

        super().__init__(
            {
                0: [
                    (TextAcceptor(value), "$") for value in enum_values
                ],
            },
        )
