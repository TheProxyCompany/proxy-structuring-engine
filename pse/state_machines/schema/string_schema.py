from __future__ import annotations

import logging
import re
from collections.abc import Callable

import regex
from pse_core import State
from pse_core.walker import Walker

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

    def get_new_walker(self, state: State | None = None) -> StringSchemaWalker:
        return StringSchemaWalker(self, state)

    def min_length(self) -> int:
        """
        Returns the minimum string length according to the schema.
        """
        return self.schema.get("minLength", 0)

    def max_length(self) -> int:
        """
        Returns the maximum string length according to the schema.
        """
        return self.schema.get("maxLength", 10000)  # Arbitrary default

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

class StringSchemaWalker(Walker):
    """
    Walker for StringSchemaAcceptor.
    """

    def __init__(
        self,
        state_machine: StringSchemaStateMachine,
        current_state: State | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: StringSchemaStateMachine = state_machine

    def should_start_transition(self, token: str) -> bool:
        if super().should_start_transition(token):
            assert self.transition_walker
            raw_value = self.transition_walker.get_raw_value()
            if self.is_within_value():
                return self.is_pattern_prefix(raw_value + token)
            return True

        return False

    def should_complete_transition(self) -> bool:
        if super().should_complete_transition():
            if self.target_state in self.state_machine.end_states:
                return self.state_machine.validate_value(self.get_current_value())
            return True

        return False

    def is_pattern_prefix(self, s: str) -> bool:
        """
        Check whether the string 's' can be a prefix of any string matching the pattern.
        """
        if self.is_within_value() and self.state_machine.pattern and self.target_state not in self.state_machine.end_states:
            pattern_str = self.state_machine.pattern.pattern
            # Use partial matching
            match = regex.match(pattern_str, s, partial=True)
            return match is not None
        return True  # If no pattern, always return True
