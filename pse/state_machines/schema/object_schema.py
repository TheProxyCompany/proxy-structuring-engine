from __future__ import annotations

from typing import Any

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.schema.key_value_schema import KeyValueSchemaStateMachine
from pse.state_machines.types.key_value import KeyValueStateMachine
from pse.state_machines.types.object import ObjectStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class ObjectSchemaStateMachine(ObjectStateMachine):
    def __init__(
        self,
        property_schema: dict[str, Any],
        context: dict[str, Any],
    ):
        self.schema = property_schema
        self.context = context
        self.properties: dict[str, Any] = property_schema.get("properties", {})
        self.required_property_names: list[str] = property_schema.get("required", [])
        self.additional_properties: dict[str, Any] | bool = property_schema.get("additionalProperties", {})
        if any(prop not in self.properties for prop in self.required_property_names):
            raise ValueError("Required property not defined in schema")

        for property_name, property_schema in self.properties.items():
            if property_name in self.required_property_names and property_schema:
                if property_schema.get("nullable", False) or "default" in property_schema:
                    self.required_property_names.remove(property_name)

        is_optional = property_schema.get("nullable", False) or not self.required_property_names
        super().__init__(is_optional)


    def get_transitions(self, stepper: Stepper) -> list[tuple[Stepper, StateId]]:
        """Retrieve transition steppers from the current state.

        Returns:
            list[tuple[Stepper, StateId]]: A list of tuples representing transitions.
        """
        value = stepper.get_current_value()
        transitions: list[tuple[Stepper, StateId]] = []
        if stepper.current_state == 2:
            for property in self.get_property_state_machines(value):
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

    def get_property_state_machines(self, value: dict[str, Any]) -> list[StateMachine]:
        property_state_machines: list[StateMachine] = []
        for prop_name, prop_schema in self.properties.items():
            if prop_name not in value:
                property = KeyValueSchemaStateMachine(
                    prop_name,
                    prop_schema,
                    self.context,
                )
                property_state_machines.append(property)

        if all(prop_name in value for prop_name in self.required_property_names) and self.additional_properties:
            # non-schema kv property to represent the additional properties
            if isinstance(self.additional_properties, dict):
                property = KeyValueSchemaStateMachine(
                    None,
                    self.additional_properties,
                    self.context,
                )
            else:
                property = KeyValueStateMachine()
            property_state_machines.append(property)

        return property_state_machines

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectSchemaStateMachine):
            return other.__eq__(self)

        return super().__eq__(other) and self.schema == other.schema

    def __str__(self) -> str:
        return super().__str__() + "Schema"
