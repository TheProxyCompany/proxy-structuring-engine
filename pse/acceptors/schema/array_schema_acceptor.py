from __future__ import annotations
from typing import Any, Dict, Type
from pse.core.walker import Walker
from pse.acceptors.collections.array_acceptor import ArrayAcceptor, ArrayWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor

class ArraySchemaAcceptor(ArrayAcceptor):

    def __init__(self, schema: Dict[str, Any], context):
        from pse.util.state_machine.get_acceptor import (
            get_acceptor,
        )

        self.schema = schema
        self.context = context
        super().__init__(
            {
                0: [
                    (TextAcceptor("["), 1),
                ],
                1: [
                    (WhitespaceAcceptor(), 2),
                    (TextAcceptor("]"), "$"),
                ],
                2: [
                    (get_acceptor(self.schema["items"], self.context), 3),
                ],
                3: [
                    (WhitespaceAcceptor(), 4),
                ],
                4: [
                    (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                    (TextAcceptor("]"), "$"),
                ],
            }
        )

    @property
    def walker_class(self) -> Type[Walker]:
        return ArraySchemaWalker

    def min_items(self) -> int:
        """
        Returns the minimum number of items in the array, according to the schema
        """
        return self.schema.get("minItems", 0)

    def max_items(self) -> int:
        """
        Returns the maximum number of items in the array, according to the schema
        """
        return self.schema.get("maxItems", 2**32)


class ArraySchemaWalker(ArrayWalker):
    """
    Walker for ArrayAcceptor
    """

    def __init__(self, acceptor: ArraySchemaAcceptor, current_state: int = 0):
        super().__init__(acceptor, current_state)
        self.acceptor = acceptor

    def should_start_transition(self, token: str) -> bool:
        if (
            self.current_state == 2
            and self.target_state == 3
            or self.current_state == 4
            and self.target_state == 2
        ):
            return len(self.value) < self.acceptor.max_items()
        if self.target_state == "$":
            return len(self.value) >= self.acceptor.min_items()

        return super().should_start_transition(token)
