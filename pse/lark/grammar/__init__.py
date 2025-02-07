from collections.abc import Callable
from dataclasses import dataclass

from lark import Lark


@dataclass
class Grammar:
    lark_grammar: Lark
    validator_function: Callable[[str, bool, str], bool]
