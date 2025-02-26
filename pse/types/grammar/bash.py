from __future__ import annotations

import logging
import os
from functools import lru_cache

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken

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
        # parser="earley",
        # ambiguity="forest",
        # lexer="dynamic",
    )


@lru_cache(maxsize=4096)
def validate_bash_code(
    parser: Lark,
    code: str,
    strict: bool = False,
    start: str = "start",
) -> bool:
    """
    Validate the Bash code.

    Args:
        parser: The Lark parser for Bash
        code: The code to validate
        strict: Whether to strictly validate the code
            If True, only accept fully valid and complete Bash code
            If False, allow code with UnexpectedEOF errors (incomplete but valid so far)
        start: The start rule of the parser

    Returns:
        Whether the code is valid
    """
    if not code or not code.strip():
        return False

    # Try different starting rules based on the code
    start_rules = [start]

    # Try each starting rule
    for rule in start_rules:
        try:
            parser.parse(code, start=rule)
            return True
        except UnexpectedEOF as e:
            # In non-strict mode, accept incomplete but valid syntax
            if not strict:
                # Only accept EOF errors, which indicate valid-so-far code
                logger.debug(f"Non-strict mode accepting incomplete code: {e}")
                return True
        except (UnexpectedCharacters, UnexpectedToken) as e:
            # These are syntax errors - continue to next rule
            logger.debug(f"Parse failed with rule {rule}: {e}")
        except Exception as e:
            # Unexpected error - continue to next rule
            logger.debug(f"Unexpected error parsing with rule {rule}: {e}")

    # All parsing attempts failed
    return False


BashGrammar = Grammar(
    name="bash",
    lark_grammar=load_bash_grammar(),
    validator_function=validate_bash_code,
    delimiters=("```bash\n", "\n```"),
)
