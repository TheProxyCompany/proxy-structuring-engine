from __future__ import annotations

from typing import Any

from pse_core import StateId
from pse_core.stepper import Stepper

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
    ):
        super().__init__(is_optional=schema.get("nullable", False))
        self.schema = schema
        self.context = context
        self.properties: dict[str, Any] = schema.get("properties", {})

        # Determine if additional properties are allowed based on the schema
        self.allow_additional_properties = (
            schema.get("additionalProperties", True) is not False
        )

        # Validate required properties
        self.required_property_names = schema.get("required", [])
        undefined_required_properties = [
            prop for prop in self.required_property_names if prop not in self.properties
        ]
        if undefined_required_properties:
            raise ValueError(
                f"Required properties not defined in schema: {', '.join(undefined_required_properties)}"
            )

    def get_transitions(self, stepper: Stepper) -> list[tuple[Stepper, StateId]]:
        """Retrieve transition steppers from the current state.

        Returns:
            list[tuple[Stepper, StateId]]: A list of tuples representing transitions.
        """
        value = stepper.get_current_value()
        transitions: list[tuple[Stepper, StateId]] = []
        if stepper.current_state == 2:
            for prop_name, prop_schema in self.properties.items():
                if prop_name not in value:
                    property = KeyValueSchemaStateMachine(
                        prop_name,
                        prop_schema,
                        self.context,
                    )
                    for transition in property.get_steppers():
                        transitions.append((transition, 3))
        elif stepper.current_state == 4:
            if all(prop_name in value for prop_name in self.required_property_names):
                for transition in PhraseStateMachine("}").get_steppers():
                    transitions.append((transition, "$"))

            if len(value) < len(self.properties):
                for transition in ChainStateMachine(
                    [PhraseStateMachine(","), WhitespaceStateMachine()]
                ).get_steppers():
                    transitions.append((transition, 2))
        else:
            return super().get_transitions(stepper)

        return transitions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectSchemaStateMachine):
            return other.__eq__(self)

        return super().__eq__(other) and self.schema == other.schema

    def __str__(self) -> str:
        return super().__str__() + "Schema"
