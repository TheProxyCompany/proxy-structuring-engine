from __future__ import annotations

import json
from typing import Any

from pse_core import StateId

from pse.state_machines import schema_to_state_machine
from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.types.key_value import KeyValueStateMachine, KeyValueStepper
from pse.state_machines.types.string import StringStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class KeyValueSchemaStateMachine(KeyValueStateMachine):
    """
    Args:
        prop_name (str): The name of the property.
        prop_schema (Dict[str, Any]): The schema of the property.
        context (Dict[str, Any]): The parsing context.
    """

    def __init__(
        self,
        prop_name: str | None,
        prop_schema: dict[str, Any],
        context: dict[str, Any],
    ):
        self.prop_name = prop_name
        self.prop_schema = prop_schema
        self.prop_context = {
            "defs": context.get("defs", {}),
            "path": f"{context.get('path', '')}/{prop_name}",
        }
        super().__init__(
            [
                (
                    PhraseStateMachine(json.dumps(self.prop_name))
                    if self.prop_name
                    else StringStateMachine()
                ),
                WhitespaceStateMachine(max_whitespace=10),
                PhraseStateMachine(":"),
                WhitespaceStateMachine(max_whitespace=10),
                schema_to_state_machine(self.prop_schema, self.prop_context),
            ],
        )

    def get_new_stepper(self, state: StateId | None = None) -> KeyValueSchemaStepper:
        return KeyValueSchemaStepper(self, state)

    @property
    def is_optional(self) -> bool:
        return super().is_optional or self.prop_schema.get("nullable", False)


class KeyValueSchemaStepper(KeyValueStepper):
    def __init__(
        self,
        state_machine: KeyValueSchemaStateMachine,
        current_state: StateId | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: KeyValueSchemaStateMachine = state_machine
