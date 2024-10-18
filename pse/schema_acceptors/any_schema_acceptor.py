from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
)
from typing import List, Dict, Any


class AnySchemaAcceptor(StateMachine):
    """
    Accepts JSON input that complies with any of several provided JSON schemas
    """

    def __init__(self, schemas: List[Dict[str, Any]], context: Dict[str, Any]) -> None:
        """
        Initialize the AnyOfAcceptor with multiple JSON schemas.

        This acceptor will validate JSON input against any of the provided schemas.

        Args:
            schemas (List[Dict[str, Any]]): A list of JSON schemas to validate against.
            context (Dict[str, Any]): Contextual information for schema definitions and paths.
        """
        from pse.util.get_acceptor import get_json_acceptor

        # Construct the state machine graph with an initial state `0` that transitions
        # to the end state `$` for each schema acceptor.

        self.acceptors = [get_json_acceptor(schema, context) for schema in schemas]
        graph: StateMachineGraph = {0: [(acceptor, "$") for acceptor in self.acceptors]}

        super().__init__(graph)
