from __future__ import annotations

import logging
from functools import lru_cache

from lark import Lark
from lark.exceptions import UnexpectedEOF, UnexpectedToken
from lark.indenter import PythonIndenter

from pse.lark.grammar import Grammar
from pse.lark.grammar.grammar import GrammarStateMachine

logger = logging.getLogger(__name__)
# only load the parser once
python_parser = Lark.open_from_package(
    "lark",
    "python.lark",
    ["grammars"],
    ambiguity="explicit",
    postlex=PythonIndenter(),
    start=["file_input"],
)


@lru_cache(maxsize=256)
def cached_parse_validation(code: str, strict: bool, start: str = "file_input") -> bool:
    """
    Attempts to parse the given code and caches the result.

    Returns True for valid (or potentially valid) code, taking into account
    non-strict mode (e.g. UnexpectedEOF or _DEDENT errors).
    """
    try:
        python_parser.parse(code, start=start)
        return True
    except Exception as e:
        # Allow incomplete input in non-strict mode.
        if isinstance(e, UnexpectedEOF) and not strict:
            return True
        elif (
            isinstance(e, UnexpectedToken)
            and not strict
            and getattr(e.token, "type", None) == "_DEDENT"
        ):
            return True
        elif not code.endswith("\n"):
            return cached_parse_validation(code + "\n", strict, start)

        return False


PythonStateMachine = GrammarStateMachine(
    Grammar(
        python_parser,
        cached_parse_validation
    )
)
