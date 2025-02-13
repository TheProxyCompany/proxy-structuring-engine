from __future__ import annotations

import json
import logging
import re

import regex
from pse_core import StateId

from pse.types.string import StringStateMachine, StringStepper

logger = logging.getLogger(__name__)


class StringSchemaStateMachine(StringStateMachine):
    """
    Accept a JSON string that conforms to a JSON schema, including 'pattern' and 'format' constraints.
    """

    # Class-level constants
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    SUPPORTED_FORMATS = frozenset(["email", "date-time", "uri"])

    def __init__(
        self,
        schema: dict,
    ):
        super().__init__(
            min_length=schema.get("minLength"),
            max_length=schema.get("maxLength"),
        )
        self.schema = schema or {}
        self.pattern: re.Pattern | None = None
        self.format: str | None = None

        if "pattern" in self.schema:
            try:
                self.pattern = re.compile(self.schema["pattern"])
            except re.error as e:
                raise ValueError(f"Invalid pattern in schema: {e}") from e

        if "format" in self.schema:
            self.format = self.schema["format"]
            if self.format not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Format '{self.format}' not supported. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                )

    def get_new_stepper(self, state: StateId | None = None) -> StringSchemaStepper:
        return StringSchemaStepper(self, state)

    def min_length(self) -> int:
        """
        Returns the minimum string length according to the schema.
        """
        return self.schema.get("minLength", 0)

    def max_length(self) -> int:
        """
        Returns the maximum string length according to the schema.
        """
        return self.schema.get("maxLength", 10000)

    def validate_email(self, value: str) -> bool:
        """
        Validate that the value is a valid email address.
        """
        return bool(self.EMAIL_PATTERN.fullmatch(value))

    def validate_date_time(self, value: str) -> bool:
        """
        Validate that the value is a valid ISO 8601 date-time.
        """
        from datetime import datetime

        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            return False

    def validate_uri(self, value: str) -> bool:
        """
        Validate that the value is a valid URI.
        """
        from urllib.parse import urlparse

        result = urlparse(value)
        return all([result.scheme, result.netloc])

    def __str__(self) -> str:
        return "JSON" + super().__str__()


class StringSchemaStepper(StringStepper):
    def __init__(
        self,
        state_machine: StringSchemaStateMachine,
        current_state: StateId | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: StringSchemaStateMachine = state_machine
        self._format_validators = {
            "email": self.state_machine.validate_email,
            "date-time": self.state_machine.validate_date_time,
            "uri": self.state_machine.validate_uri,
        }

    def should_start_step(self, token: str) -> bool:
        if super().should_start_step(token):
            if self.is_within_value():
                valid_prefix = self.get_valid_prefix(token)
                return self.validate_value(valid_prefix)
            return True

        return False

    def consume(self, token: str):
        """
        Consume the token and return the new stepper.
        """
        if self.is_within_value():
            valid_prefix = self.get_valid_prefix(token)
            if not valid_prefix:
                return []
        else:
            valid_prefix = token

        steppers = super().consume(valid_prefix)
        for stepper in steppers:
            if token != valid_prefix:
                stepper.remaining_input = token[len(valid_prefix) :]

        return steppers

    def clean_value(self, value: str) -> str:
        """
        Clean and normalize the input value.

        Args:
            value: The string value to clean

        Returns:
            str: The cleaned value

        Raises:
            ValueError: If the value cannot be properly cleaned
        """
        try:
            if value.startswith('"') and value.endswith('"'):
                return json.loads(value) or ""
            elif value.startswith('"'):
                value = value[value.index('"') + 1 :]
                return value[: value.rindex('"')] if '"' in value else value
            return value
        except Exception as e:
            logger.debug(f"Error cleaning value: {e}")
            return value

    def get_valid_prefix(self, s: str) -> str | None:
        """
        Check whether the string 's' can be a prefix of any string matching the pattern.
        """
        if (
            not self.is_within_value()
            or not self.state_machine.pattern
            or not self.sub_stepper
        ):
            return s

        match = None
        s = self.clean_value(s)
        current_value = self.sub_stepper.get_raw_value()
        while not match and s:
            match = regex.match(
                self.state_machine.pattern.pattern,
                current_value + s,
                partial=True,
            )
            if not match:
                s = s[:-1]

        return s if match else None

    def validate_value(self, value: str | None = None) -> bool:
        """
        Validate the string value according to the schema.

        Args:
            value: Optional string to append to current value before validation

        Returns:
            bool: True if the value meets all schema constraints

        Note:
            Validates length, pattern, and format constraints in sequence
        """
        value = self.clean_value(self.get_raw_value() + (value or ""))
        if not value:
            return False

        # Length validation
        if len(value) > self.state_machine.max_length():
            return False

        # Pattern validation
        if not self.is_within_value() and self.state_machine.pattern:
            if not self.state_machine.pattern.match(value):
                return False

        # Format validation
        if self.state_machine.format:
            validator = self._format_validators.get(self.state_machine.format)
            if not validator:
                raise ValueError(
                    f"No validator found for format: {self.state_machine.format}"
                )
            if not validator(value):
                return False

        return True
