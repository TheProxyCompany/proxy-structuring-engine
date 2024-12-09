from typing import Any

from pse_core.state_machine import StateMachine


class AnySchemaAcceptor(StateMachine):
    """
    Accepts JSON input that complies with any of several provided JSON schemas
    """

    def __init__(self, schemas: list[dict[str, Any]], context: dict[str, Any]) -> None:
        """
        Initialize the AnyOfAcceptor with multiple JSON schemas.

        This state_machine will validate JSON input against any of the provided schemas.

        Args:
            schemas (List[Dict[str, Any]]): A list of JSON schemas to validate against.
            context (Dict[str, Any]): Contextual information for schema definitions and paths.
        """
        from pse.util.get_state_machine import get_state_machine

        # Construct the state machine graph with an initial state `0` that transitions
        # to the end state `$` for each schema state_machine.
        self.acceptors = [get_state_machine(schema, context) for schema in schemas]

        super().__init__(
            {0: [(state_machine, "$") for state_machine in self.acceptors]}
        )
