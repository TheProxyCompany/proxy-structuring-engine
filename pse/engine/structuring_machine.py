import logging

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.base.any import AnyStateMachine
from pse.base.encapsulated import EncapsulatedStateMachine
from pse.base.wait_for import WaitFor
from pse.grammar.grammar import GrammarStateMachine
from pse.grammar.python import PythonGrammar
from pse.json import (
    JSONSchemaSource,
    generate_json_schema,
    json_schema_to_state_machine,
)

logger = logging.getLogger(__name__)

class StructuringMachine(StateMachine):

    def __init__(
        self,
        json_schemable: JSONSchemaSource,
        delimiters: tuple[str, str] | None = None,
        min_buffer_length: int = -1,
        include_python: bool = False,
    ):
        state_machines = []
        if include_python:
            python_state_machine = EncapsulatedStateMachine(
                GrammarStateMachine(PythonGrammar),
                delimiters=PythonGrammar.delimiters,
                min_buffer_length=0,
            )
            state_machines.append(python_state_machine)

        if json_schemable:
            json_schema = generate_json_schema(json_schemable)
            state_machine = json_schema_to_state_machine(json_schema)
            if delimiters:
                state_machines.append(
                    EncapsulatedStateMachine(
                        state_machine,
                        delimiters,
                        min_buffer_length,
                    )
                )
            elif min_buffer_length >= 0:
                state_machines.append(WaitFor(state_machine, min_buffer_length))
            else:
                state_machines.append(state_machine)

        super().__init__(
            {
                0: [
                    (AnyStateMachine(state_machines), "$"),
                ]
            }
        )

    def get_steppers(self, state: StateId | None = None) -> list[Stepper]:
        steppers = []
        for edge, _ in self.get_edges(state or 0):
            steppers.extend(edge.get_steppers())
        return steppers
