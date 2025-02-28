from __future__ import annotations

import logging
import os

from lark import Lark

from pse.types.grammar import Grammar

logger = logging.getLogger(__name__)


def load_bash_grammar():
    """
    Load the Bash grammar from the Lark file.
    """
    # Get the path to the bash.lark file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    grammar_path = os.path.join(current_dir, "bash.lark")

    # Read the Lark file
    with open(grammar_path) as f:
        bash_grammar_content = f.read()

    return Lark(
        bash_grammar_content,
        start="start",
        parser="lalr",
        # ambiguity="forest",
        lexer="dynamic-complete",
    )

def validate_bash_code(
    parser: Lark,
    code: str,
    strict: bool = False,
    start: str = "start",
) -> bool:
    return True


BashGrammar = Grammar(
    name="bash",
    lark_grammar=load_bash_grammar(),
    delimiters=("```bash\n", "\n```"),
    validator_function=validate_bash_code,
)
