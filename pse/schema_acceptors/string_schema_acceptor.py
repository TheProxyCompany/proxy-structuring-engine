from __future__ import annotations
import json
import re
from typing import Callable, Optional, Any

from pse.util.errors import SchemaNotImplementedError
from pse.acceptors.json.string_acceptor import StringAcceptor, StringWalker

import logging
import regex  # Note: This is a third-party module

logger = logging.getLogger(__name__)


class StringSchemaAcceptor(StringAcceptor):
    """
    Accept a JSON string that conforms to a JSON schema, including 'pattern' and 'format' constraints.
    """

    def __init__(
        self,
        schema: dict,
        start_hook: Optional[Callable] = None,
        end_hook: Optional[Callable] = None,
    ):
        super().__init__(StringSchemaWalker)
        self.schema = schema or {}
        self.start_hook = start_hook
        self.end_hook = end_hook

        self.pattern: Optional[re.Pattern] = None
        self.format: Optional[str] = None

        if "pattern" in self.schema:
            pattern_str = self.schema["pattern"]
            self.pattern = re.compile(pattern_str)
        if "format" in self.schema:
            self.format = self.schema["format"]
            # support 'email', 'date-time', 'uri' formats
            if self.format not in ["email", "date-time", "uri"]:
                raise SchemaNotImplementedError(
                    f"Format '{self.format}' not implemented"
                )

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
                raise SchemaNotImplementedError(
                    f"Format '{self.format}' not implemented"
                )
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


class StringSchemaWalker(StringWalker):
    """
    Walker for StringSchemaAcceptor.
    """

    def __init__(self, acceptor: StringSchemaAcceptor, current_state: int = 0):
        super().__init__(acceptor, current_state)
        self.acceptor = acceptor
        self.is_escaping = False

    def should_complete_transition(self) -> bool:
        in_string_content = self.is_in_string_content()
        if (
            not in_string_content
            and self.target_state == self.acceptor.STATE_IN_STRING
            and self.acceptor.start_hook
        ):
            self.acceptor.start_hook()

        # Only update partial_value when processing actual string content
        if in_string_content and self.target_state and self.target_state not in self.acceptor.end_states:
            if self.is_escaping:
                self.is_escaping = False
            elif self.transition_walker and self.transition_walker.raw_value == "\\":
                self.is_escaping = True

            if (
                self.acceptor.pattern
                and self._raw_value
                and not self.is_pattern_prefix(self._raw_value)
            ):
                return False  # Reject early if pattern can't match

        if self.target_state and self.target_state in self.acceptor.end_states:
            if self.acceptor.end_hook:
                self.acceptor.end_hook()

            if self.acceptor.validate_value(self.get_current_value()):
                return True
            else:
                return False

        return True

    def is_in_string_content(self) -> bool:
        """
        Determine if the walker is currently inside the string content (i.e., after the opening quote).
        """
        return self.current_state == self.acceptor.STATE_IN_STRING

    def is_pattern_prefix(self, s: str) -> bool:
        """
        Check whether the string 's' can be a prefix of any string matching the pattern.
        """
        if self.acceptor.pattern:
            pattern_str = self.acceptor.pattern.pattern
            # Use partial matching
            match = regex.match(pattern_str, s, partial=True)
            return match is not None
        return True  # If no pattern, always return True

    def get_current_value(self) -> Any:
        return json.loads(self.raw_value) if self.raw_value else None
