from __future__ import annotations
from pse.acceptors.basic.number.number_acceptor import NumberAcceptor, NumberWalker
from typing import Type


class NumberSchemaAcceptor(NumberAcceptor):
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

    def validate_value(self, value):
        """
        Validate the number value according to the schema
        """
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

        if self.is_integer and not isinstance(value, int):
            return False
        return True

    @property
    def walker_class(self) -> Type[NumberSchemawalker]:
        return NumberSchemawalker


class NumberSchemawalker(NumberWalker):
    """
    Walker for NumberAcceptor
    """

    def __init__(self, acceptor: NumberSchemaAcceptor):
        super().__init__(acceptor)
        self.acceptor = acceptor

    def should_start_transition(self, transition_acceptor):
        if (
            self.acceptor.is_integer
            and self.current_state == 3
            and self.target_state == 4
        ):
            return False
        return super().should_start_transition(transition_acceptor)

    def should_complete_transition(self, transition_value, is_end_state) -> bool:
        if not super().should_complete_transition(transition_value, is_end_state):
            return False
        # Only validate when there is no remaining input
        if is_end_state and not self.remaining_input:
            return self.acceptor.validate_value(self.current_value())
        return True
