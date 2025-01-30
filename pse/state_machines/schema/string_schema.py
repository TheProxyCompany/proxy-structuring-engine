from __future__ import annotations

import logging
import re
from collections.abc import Callable

import regex
from pse_core import StateId
from pse_core.stepper import Stepper

from pse.state_machines.types.string import StringStateMachine

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

    def validate_value(self, value: str) -> bool:
        """
        Validate the string value according to the schema.
        """
        if len(value) < self.min_length():
            return False
        if len(value) > self.max_length():
            return False
        if self.pattern and not self.pattern.fullmatch(value):
            return False
        if self.format:
            format_validator = {
                "email": self.validate_email,
                "date-time": self.validate_date_time,
                "uri": self.validate_uri,
            }.get(self.format)
            if format_validator and not format_validator(value):
                return False
            elif not format_validator:
                raise ValueError(f"Format '{self.format}' not implemented")
        return True

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


class StringSchemaStepper(Stepper):
    def __init__(
        self,
        state_machine: StringSchemaStateMachine,
        current_state: StateId | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: StringSchemaStateMachine = state_machine

    def should_start_step(self, token: str) -> bool:
        if super().should_start_step(token):
            assert self.sub_stepper
            raw_value = self.sub_stepper.get_raw_value()
            if self.is_within_value():
                valid_prefix = self.valid_prefix(raw_value + token)
                return valid_prefix is not None and raw_value != valid_prefix
            return True

        return False

    def should_complete_step(self) -> bool:
        if super().should_complete_step():
            if self.target_state in self.state_machine.end_states:
                return self.state_machine.validate_value(self.get_current_value())
            return True

        return False

    def valid_prefix(self, s: str) -> str | None:
        """
        Check whether the string 's' can be a prefix of any string matching the pattern.
        """
        if not self.is_within_value() or not self.state_machine.pattern:
            return s

        match = None
        pattern_str = self.state_machine.pattern.pattern
        while not match and s:
            match = regex.match(pattern_str, s, partial=True)
            if not match:
                s = s[:-1]

        return s if match else None
