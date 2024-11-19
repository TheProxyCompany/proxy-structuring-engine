from __future__ import annotations
from typing import Type
from pse.acceptors.basic.number_acceptor import NumberAcceptor, NumberWalker
from pse.core.walker import Walker

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

    @property
    def walker_class(self) -> Type[Walker]:
        return NumberSchemaWalker

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


class NumberSchemaWalker(NumberWalker):
    """
    Walker for NumberAcceptor
    """

    def __init__(self, acceptor: NumberSchemaAcceptor, current_state: int = 0):
        super().__init__(acceptor, current_state)
        self.acceptor = acceptor

    def should_start_transition(self, token: str) -> bool:
        if self.acceptor.is_integer and self.target_state == 3:
            return False
        return super().should_start_transition(token)

    def should_complete_transition(self) -> bool:
        if not super().should_complete_transition():
            return False
        # Only validate when there is no remaining input
        if (
            self.target_state is not None
            and self.target_state in self.acceptor.end_states
            and not self.remaining_input
        ):
            return self.acceptor.validate_value(self.current_value)
        return True
