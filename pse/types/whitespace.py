"""Whitespace state machine for parsing optional whitespace in structured data.

This module provides a state machine for recognizing and parsing whitespace
characters in structured data formats like JSON.
"""

from __future__ import annotations

import logging

from pse.types.base.character import CharacterStateMachine

logger = logging.getLogger()

# Whitespace characters as defined by the JSON standard
WHITESPACE_CHARS = " \t\n\r"


class WhitespaceStateMachine(CharacterStateMachine):
    """Optional whitespace state machine using TokenTrie for efficient matching."""

    def __init__(self, min_whitespace: int = 0, max_whitespace: int = 20):
        """Initialize the whitespace state machine with configurable limits.

        Args:
            min_whitespace: Minimum allowable whitespace characters.
                Defaults to 0.
            max_whitespace: Maximum allowable whitespace characters.
                Defaults to 20.
        """
        super().__init__(
            WHITESPACE_CHARS,
            char_min=min_whitespace,
            char_limit=max_whitespace,
            is_optional=(min_whitespace == 0),
        )

    def __str__(self) -> str:
        """Return a string representation of this state machine."""
        return "Whitespace"
