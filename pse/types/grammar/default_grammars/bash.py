from __future__ import annotations

import logging
import os

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken

from pse.types.grammar import LarkGrammar

logger = logging.getLogger(__name__)


class BashGrammar(LarkGrammar):

    def __init__(self):
        # Get the path to the bash.lark file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        grammar_path = os.path.join(current_dir, "bash.lark")

        # Read the Lark file
        with open(grammar_path) as f:
            bash_grammar_content = f.read()

        bash_lark_grammar = Lark(
            bash_grammar_content,
            start="start",
            parser="lalr",
            lexer="basic",
        )

        super().__init__(
            name="Bash",
            lark_grammar=bash_lark_grammar,
            delimiters=("```bash\n", "\n```"),
        )

    def validate(
        self,
        input: str,
        strict: bool = False,
        start: str | None = None,
    ) -> bool:
        """
        Validate Bash code using the Lark parser.

        Args:
            input: The Bash code to validate.
            strict: Whether to use strict validation.
            start: The start rule to use.
        """
        # If code is empty, it's not valid bash
        if not input.strip():
            return False

        try:
            # Try to parse the input normally
            self.lark_grammar.parse(input, start=start or "start")
            return True
        except Exception as e:
            # If we're in strict mode, any parse error means invalid
            if strict:
                return False

            # EOF errors are typically valid incomplete syntax
            if isinstance(e, UnexpectedEOF):
                # Token errors require more analysis
                return True
            elif isinstance(e, UnexpectedToken):
                # End of input is generally fine in non-strict mode
                if e.token.type == "$END":
                    return True
            elif isinstance(e, UnexpectedCharacters):
                return True

            # Default is to reject
            return False
