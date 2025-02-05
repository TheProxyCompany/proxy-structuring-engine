from typing import Any

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper


class AnySchemaStateMachine(StateMachine):
    """
    Accepts JSON input that complies with any of several provided JSON schemas
    """

    def __init__(self, schemas: list[dict[str, Any]], context: dict[str, Any]) -> None:
        """
        This state_machine will validate JSON input against any of the provided schemas.

        Args:
            schemas (List[Dict[str, Any]]): A list of JSON schemas to validate against.
            context (Dict[str, Any]): Contextual information for schema definitions and paths.
        """
        from pse.state_machines import schema_to_state_machine

        # Construct the state machine graph with an initial state `0` that transitions
        # to the end state `$` for each schema state_machine.
        self.state_machines: list[StateMachine] = []
        for schema in schemas:
            sm = schema_to_state_machine(schema, context)
            self.state_machines.append(sm)

        super().__init__(
            {
                0: [
                    (state_machine, "$")
                    for state_machine in self.state_machines
                ]
            }
        )

    def get_steppers(self, state: StateId | None = None) -> list[Stepper]:
        steppers = []
        for edge, _ in self.get_edges(state or 0):
            steppers.extend(edge.get_steppers())
        return steppers

    def __str__(self) -> str:
        return "AnyStateMachine"
