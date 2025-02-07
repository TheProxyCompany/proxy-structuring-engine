import logging

from pse_core.state_machine import StateMachine

from pse.base.encapsulated import EncapsulatedStateMachine
from pse.base.wait_for import WaitFor
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
) -> StateMachine:
    """
    Build a state_machine based on the provided schema.

    Args:
        json_schemable (JSONSchemaSource): The schema to validate against.
        delimiters (tuple[str, str] | None): The delimiters to indicate the start and end of the schema.
        buffer_length (int): The buffer size before enforcing the schema.

    Returns:
        StateMachine: An state_machine based on the schema.
    """
    json_schema = generate_json_schema(json_schemable)
    state_machine = json_schema_to_state_machine(json_schema)
    if delimiters:
        return EncapsulatedStateMachine(
            state_machine,
            delimiters,
            min_buffer_length,
        )
    elif min_buffer_length >= 0:
        return WaitFor(state_machine, min_buffer_length)
    else:
        return state_machine
