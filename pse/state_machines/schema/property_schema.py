from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pse_core import State

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.get_state_machine import get_state_machine
from pse.state_machines.types.property import PropertyStateMachine, PropertyWalker
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class PropertySchemaStateMachine(PropertyStateMachine):
    """
    Acceptor for an object property according to the schema.

    Args:
        prop_name (str): The name of the property.
        prop_schema (Dict[str, Any]): The schema of the property.
        context (Dict[str, Any]): The parsing context.
        value_started_hook (Callable | None, optional): Hook called when value parsing starts.
        value_ended_hook (Callable | None, optional): Hook called when value parsing ends.
    """

    def __init__(
        self,
        prop_name: str,
        prop_schema: dict[str, Any],
        context: dict[str, Any],
        value_started_hook: Callable | None = None,
        value_ended_hook: Callable | None = None,
    ):
        self.prop_name = prop_name
        self.prop_schema = prop_schema
        self.prop_context = {
            "defs": context.get("defs", {}),
            "path": f"{context.get('path', '')}/{prop_name}",
        }
        super().__init__(
            [
                PhraseStateMachine(json.dumps(self.prop_name)),
                WhitespaceStateMachine(),
                PhraseStateMachine(":"),
                WhitespaceStateMachine(),
                get_state_machine(
                    self.prop_schema,
                    self.prop_context,
                    value_started_hook,
                    value_ended_hook,
                ),
            ],
        )

    def get_new_walker(self, state: State | None = None) -> PropertySchemaWalker:
        return PropertySchemaWalker(self, state)

    @property
    def is_optional(self) -> bool:
        return super().is_optional or self.prop_schema.get("nullable", False)


class PropertySchemaWalker(PropertyWalker):
    """
    Walker for PropertySchemaAcceptor
    """

    def __init__(
        self,
        state_machine: PropertySchemaStateMachine,
        current_state: State | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: PropertySchemaStateMachine = state_machine

    def should_complete_transition(self) -> bool:
        if not super().should_complete_transition():
            return False

        hooks: dict[str, Callable] = self.state_machine.prop_schema.get("__hooks", {})
        prop_name = self.state_machine.prop_name
        if self.target_state == 4:
            if "value_start" in hooks:
                hooks["value_start"](prop_name)
        elif self.target_state and self.target_state in self.state_machine.end_states:
            if "value_end" in hooks:
                hooks["value_end"](prop_name, self.prop_value)
        return True

    def get_current_value(self):
        return (self.state_machine.prop_name, self.prop_value)
