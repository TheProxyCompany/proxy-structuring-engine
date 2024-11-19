from enum import Enum
from typing import Tuple


class DelimiterType(Enum):
    """
    Represents the type of structured schema constraints for output.

    Attributes:
        JSON (DelimiterType): Output constrained by a JSON schema.
        PYTHON (DelimiterType): Output constrained by a Python schema.
        CYPHER (DelimiterType): Output constrained by a Cypher schema.
        U_DIFF (DelimiterType): Output constrained by a U_DIFF schema.
    """

    JSON = (("```json\n", "\n```"),)
    PYTHON = (("```python\n", "\n```"),)
    CYPHER = (("```cypher\n", "\n```"),)
    U_DIFF = (("```diff\n", "\n```"),)

    def __init__(self, delimiters: Tuple[str, str]) -> None:
        self._delimiters = delimiters

    @property
    def delimiters(self) -> Tuple[str, str]:
        """
        Returns the opening and closing delimiters based on the DelimiterType.

        Returns:
            Tuple[str, str]: A tuple containing the opening and closing delimiters.
        """
        return self._delimiters
