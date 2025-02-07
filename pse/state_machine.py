import logging

from pse_core.state_machine import StateMachine

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


def build_state_machine(
    json_schemable: JSONSchemaSource,
    delimiters: tuple[str, str] | None = None,
    min_buffer_length: int = -1,
    include_python: bool = False,
) -> StateMachine:
    """
    Build a state_machine based on the provided schema.

    Args:
        json_schemable (JSONSchemaSource): The schema to validate against.
        delimiters (tuple[str, str] | None): The delimiters to indicate the start and end of the schema.
        buffer_length (int): The buffer size before enforcing the schema.
        include_python (bool): Whether to include a Python Interpreter state_machine.

    Returns:
        StateMachine: An state_machine based on the schema.
    """
    state_machines = []
    if include_python:
        python_state_machine = EncapsulatedStateMachine(
            GrammarStateMachine(PythonGrammar),
            delimiters=("```python\n", "\n```"),
            min_buffer_length=min_buffer_length,
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

    if len(state_machines) > 1:
        return AnyStateMachine(state_machines)
    else:
        return state_machines[0]
