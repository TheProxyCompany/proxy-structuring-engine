from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

import regex
from pse_core import StateId

from pse.state_machines.types.string import StringStateMachine, StringStepper

logger = logging.getLogger(__name__)


class StringSchemaStateMachine(StringStateMachine):
    """
    Accept a JSON string that conforms to a JSON schema, including 'pattern' and 'format' constraints.
    """

    def __init__(
        self,
        schema: dict,
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
    ):
        super().__init__()
        self.schema = schema or {}
        self.start_hook = start_hook
        self.end_hook = end_hook

        self.pattern: re.Pattern | None = None
        self.format: str | None = None

        if "pattern" in self.schema:
            pattern_str = self.schema["pattern"]
            self.pattern = re.compile(pattern_str)
        if "format" in self.schema:
            self.format = self.schema["format"]
            # support 'email', 'date-time', 'uri' formats
            if self.format not in ["email", "date-time", "uri"]:
                raise ValueError(f"Format '{self.format}' not implemented")

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
        email_regex = re.compile(r"[^@]+@[^@]+\.[^@]+")
        return email_regex.fullmatch(value) is not None

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
        return super().__str__() + "Schema"


class StringSchemaStepper(StringStepper):
    def __init__(
        self,
        state_machine: StringSchemaStateMachine,
        current_state: StateId | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: StringSchemaStateMachine = state_machine

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
        if value.startswith('"') and value.endswith('"'):
            try:
                value = json.loads(value) or ""
            except Exception:
                return ""
        elif value.startswith('"'):
            value = value[value.index('"') + 1:]
            if '"' in value:
                value = value[:value.rindex('"')]

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
        """
        # Extract content before the last quote if present, otherwise use as-is
        value = self.clean_value(self.get_raw_value() + (value or ""))
        if not value:
            return False

        if len(value) < self.state_machine.min_length():
            return False
        if len(value) > self.state_machine.max_length():
            return False

        if not self.is_within_value() and self.state_machine.pattern:
            return self.state_machine.pattern.match(value) is not None

        if self.state_machine.format:
            format_validator = {
                "email": self.state_machine.validate_email,
                "date-time": self.state_machine.validate_date_time,
                "uri": self.state_machine.validate_uri,
            }.get(self.state_machine.format)
            if format_validator and not format_validator(value):
                return False
            elif not format_validator:
                raise ValueError(f"Format '{self.state_machine.format}' not implemented")
        return True
