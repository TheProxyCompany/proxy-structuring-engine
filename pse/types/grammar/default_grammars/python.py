from __future__ import annotations

import logging
import os

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from lark.indenter import PythonIndenter

from pse.types.grammar import LarkGrammar

logger = logging.getLogger(__name__)

class PythonGrammar(LarkGrammar):

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        grammar_path = os.path.join(current_dir, "python.lark")

        # Read the Lark file
        with open(grammar_path) as f:
            python_grammar_content = f.read()
        python_lark_grammar = Lark(
            python_grammar_content,
            parser="lalr",
            lexer="basic",
            postlex=PythonIndenter(),
            start=["file_input"],
        )

        super().__init__(
            name="Python",
            lark_grammar=python_lark_grammar,
            delimiters=("```python\n", "\n```"),
        )

    def validate(
        self,
        input: str,
        strict: bool = False,
        start: str | None = None,
    ) -> bool:
        """
        Validate Python code using the Lark parser.

        Args:
            parser: The Lark parser to use.
            code: The Python code to validate.
            strict: Whether to use strict validation.
        """
        if strict and not input.endswith("\n"):
            input += "\n"

        try:
            self.lark_grammar.parse(input, start=start)
            return True
        except Exception as e:
            if not strict and isinstance(e, UnexpectedToken):
                return e.token.type == "_DEDENT" or e.token.type == "$END"
            if not strict and isinstance(e, UnexpectedCharacters):
                # special case for unclosed quotes
                return e.char == "'" or e.char == '"'

            return False
