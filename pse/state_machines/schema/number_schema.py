from __future__ import annotations

from pse_core import StateId
from pse_core.stepper import Stepper

from pse.state_machines.types.number import NumberStateMachine


class NumberSchemaStateMachine(NumberStateMachine):
    """
    Accept a JSON number that conforms to a JSON schema
    """

    def __init__(self, schema):
        super().__init__()
        self.schema = schema
        self.is_integer = schema["type"] == "integer"
        self.requires_validation = any(
            constraint in schema
            for constraint in [
                "minimum",
                "exclusiveMinimum",
                "maximum",
                "exclusiveMaximum",
                "multipleOf",
            ]
        )

    def get_new_stepper(self, state: StateId | None = None) -> NumberSchemaStepper:
        return NumberSchemaStepper(self, state)

    def validate_value(self, value: float) -> bool:
        """
        Validate the number value according to the schema
        """
        if not isinstance(value, int | float):
            return True

        if "minimum" in self.schema and value < self.schema["minimum"]:
            return False

        if (
            "exclusiveMinimum" in self.schema
            and value <= self.schema["exclusiveMinimum"]
        ):
            return False
        if "maximum" in self.schema and value > self.schema["maximum"]:
            return False
        if (
            "exclusiveMaximum" in self.schema
            and value >= self.schema["exclusiveMaximum"]
        ):
            return False
        if "multipleOf" in self.schema:
            divisor = self.schema["multipleOf"]
            if value / divisor != value // divisor:
                return False

        if self.is_integer and not value.is_integer():
            return False

        return True

    def __str__(self) -> str:
        return super().__str__() + "Schema"


class NumberSchemaStepper(Stepper):
    """ """

    def __init__(
        self,
        state_machine: NumberSchemaStateMachine,
        current_state: StateId | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: NumberSchemaStateMachine = state_machine

    def should_start_step(self, token: str) -> bool:
        if self.state_machine.is_integer and self.target_state == 3:
            return False

        return super().should_start_step(token)

    def should_complete_step(self) -> bool:
        if not super().should_complete_step():
            return False

        return self.state_machine.validate_value(self.get_current_value())

    def has_reached_accept_state(self) -> bool:
        if super().has_reached_accept_state():
            return self.state_machine.validate_value(self.get_current_value())

        return False
