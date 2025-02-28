from __future__ import annotations

import logging

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken
from lark.indenter import PythonIndenter

from pse.types.grammar import LarkGrammar

logger = logging.getLogger(__name__)

class PythonGrammar(LarkGrammar):

    def __init__(self):
        python_lark_grammar = Lark.open_from_package(
            "lark",
            "python.lark",
            ["grammars"],
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
            if not strict:
                # Handle incomplete strings and other incomplete constructs
                if isinstance(e, UnexpectedEOF | UnexpectedCharacters):
                    return True
                elif isinstance(e, UnexpectedToken) and (
                    e.token.type == "_DEDENT" or e.token.type == "$END"
                ):
                    return True
                elif not input.endswith("\n"):
                    return self.validate(input, True)

            return False
