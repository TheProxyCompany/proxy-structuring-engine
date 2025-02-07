from __future__ import annotations

import logging
from functools import lru_cache

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken
from lark.indenter import PythonIndenter

from pse.grammar import Grammar

logger = logging.getLogger(__name__)
# only load the parser once
python_parser = Lark.open_from_package(
    "lark",
    "python.lark",
    ["grammars"],
    ambiguity="resolve",
    postlex=PythonIndenter(),
    start=["file_input"],
)


@lru_cache(maxsize=4096)
def validate_python_code(
    parser: Lark,
    code: str,
    strict: bool = False,
    start: str = "file_input",
) -> bool:
    if strict and not code.endswith("\n"):
        return validate_python_code(parser, code + "\n", strict, start)

    try:
        parser.parse(code, start=start)
        return True
    except Exception as e:
        if not strict:
            # Handle incomplete strings and other incomplete constructs
            if isinstance(e, UnexpectedEOF | UnexpectedCharacters):
                return True
            elif (
                isinstance(e, UnexpectedToken)
                and getattr(e.token, "type", None) == "_DEDENT"
            ):
                return True
            elif not code.endswith("\n"):
                return validate_python_code(parser, code + "\n", strict, start)

        return False


PythonGrammar = Grammar(
    name="Python",
    lark_grammar=python_parser,
    validator_function=validate_python_code,
    delimiters=("```python\n", "\n```"),
)
