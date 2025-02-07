from collections.abc import Callable
from dataclasses import dataclass

from lark import Lark


@dataclass(frozen=True, slots=True)
class Grammar:

    name: str
    lark_grammar: Lark
    validator_function: Callable[[Lark, str, bool, str], bool]
    delimiters: tuple[str, str]

    def validate(
        self,
        input: str,
        strict: bool = False,
        start: str = "file_input",
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
        return self.validator_function(self.lark_grammar, input, strict, start)
