# from __future__ import annotations

# import logging
# import os
# from functools import lru_cache

# from lark import Lark
# from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken

# from pse.types.grammar import Grammar

# logger = logging.getLogger(__name__)


# def load_bash_grammar():
#     """
#     Load the Bash grammar from the Lark file.
#     """
#     # Get the path to the bash.lark file
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     grammar_path = os.path.join(current_dir, "bash.lark")

#     # Read the Lark file
#     with open(grammar_path) as f:
#         bash_grammar_content = f.read()

#     return Lark(
#         bash_grammar_content,
#         start="start",
#         parser="earley",
#         lexer="dynamic_complete",
#     )

# @lru_cache(maxsize=4096)
# def validate_bash_code(
#     parser: Lark,
#     code: str,
#     strict: bool = False,
#     start: str = "start",
# ) -> bool:
#     """
#     Validate Python code using the Lark parser.

#     Args:
#         parser: The Lark parser to use.
#         code: The Python code to validate.
#         strict: Whether to use strict validation.
#     """
#     try:
#         parser.parse(code, start=start)
#         return True
#     except Exception as e:
#         if not strict:
#             # Handle incomplete strings and other incomplete constructs
#             if isinstance(e, UnexpectedEOF | UnexpectedCharacters):
#                 return True
#             elif isinstance(e, UnexpectedToken) and (
#                 e.token.type == "DONE"
#             ):
#                 return True

#         return False


# BashGrammar = Grammar(
#     name="bash",
#     lark_grammar=load_bash_grammar(),
#     delimiters=("```bash\n", "\n```"),
#     validator_function=validate_bash_code,
# )
