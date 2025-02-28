from abc import ABC, abstractmethod

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken


class LarkGrammar(ABC):

    name: str
    lark_grammar: Lark
    delimiters: tuple[str, str] | None = None

    def __init__(
        self,
        name: str,
        lark_grammar: Lark,
        delimiters: tuple[str, str] | None = None,
    ):
        self.name = name
        self.lark_grammar = lark_grammar
        self.delimiters = delimiters

    @abstractmethod
    def validate(
        self,
        input: str,
        strict: bool = False,
        start: str | None = None,
    ) -> bool:
        """
        Validate the input against the grammar.

        Args:
            input (str): The input to validate.
            strict (bool): Whether to use strict validation.
            start (str): The start rule to use.

        Returns:
            bool: True if the input is valid, False otherwise.
        """
        try:
            self.lark_grammar.parse(input, start=start)
            return True
        except Exception as e:
            if not strict:
                if isinstance(e, UnexpectedEOF | UnexpectedCharacters):
                    return True
                elif isinstance(e, UnexpectedToken) and e.token.type == "$END":
                    return True

            return False

from pse.types.grammar.default_grammars.bash import BashGrammar  # noqa: E402
from pse.types.grammar.default_grammars.python import PythonGrammar  # noqa: E402
from pse.types.grammar.lark import LarkGrammarStateMachine  # noqa: E402

PythonStateMachine = LarkGrammarStateMachine(PythonGrammar())
BashStateMachine = LarkGrammarStateMachine(BashGrammar())

__all__ = [
    "BashStateMachine",
    "LarkGrammarStateMachine",
    "PythonStateMachine",
]
