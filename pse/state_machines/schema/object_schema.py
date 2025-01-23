from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pse_core import State
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.schema.key_value_schema import KeyValueSchemaStateMachine
from pse.state_machines.types.object import ObjectStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class ObjectSchemaStateMachine(ObjectStateMachine):
    def __init__(
        self,
        schema: dict[str, Any],
        context: dict[str, Any],
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
    ):
        super().__init__()
        self.schema = schema
        self.context = context
        self.properties: dict[str, Any] = schema.get("properties", {})
        self.start_hook = start_hook
        self.end_hook = end_hook

        # Determine if additional properties are allowed based on the schema
        self.allow_additional_properties = schema.get("additionalProperties", True) is not False

        # Validate required properties
        self.required_property_names = schema.get("required", [])
        undefined_required_properties = [
            prop for prop in self.required_property_names if prop not in self.properties
        ]
        if undefined_required_properties:
            raise ValueError(
                f"Required properties not defined in schema: {', '.join(undefined_required_properties)}"
            )

    def get_transitions(self, walker: Walker) -> list[tuple[Walker, State]]:
        """Retrieve transition walkers from the current state.

        For each edge from the current state, returns walkers that can traverse that edge.
        Handles optional acceptors and pass-through transitions appropriately.

        Args:
            walker: The walker initiating the transition.
            state: Optional starting state. If None, uses the walker's current state.

        Returns:
            list[tuple[Walker, State]]: A list of tuples representing transitions.
        """
        value = walker.get_current_value()
        transitions: list[tuple[Walker, State]] = []
        if walker.current_state == 2:
            for prop_name, prop_schema in self.properties.items():
                if prop_name not in value:
                    property = KeyValueSchemaStateMachine(
                        prop_name,
                        prop_schema,
                        self.context,
                        self.start_hook,
                        self.end_hook,
                    )
                    for transition in property.get_walkers():
                        transitions.append((transition, 3))
        elif walker.current_state == 4:
            if all(prop_name in value for prop_name in self.required_property_names):
                for transition in PhraseStateMachine("}").get_walkers():
                    transitions.append((transition, "$"))

            if len(value) < len(self.properties):
                for transition in ChainStateMachine([PhraseStateMachine(","), WhitespaceStateMachine()]).get_walkers():
                    transitions.append((transition, 2))
        else:
            return super().get_transitions(walker)

        return transitions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectSchemaStateMachine):
            return other.__eq__(self)

        return super().__eq__(other) and self.schema == other.schema

    def __str__(self) -> str:
        return super().__str__() + "Schema"
