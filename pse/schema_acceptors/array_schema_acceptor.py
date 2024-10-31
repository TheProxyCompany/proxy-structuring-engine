from __future__ import annotations
from typing import Any, Dict, Type
from pse.acceptors.collections.array_acceptor import ArrayAcceptor, ArrayWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.state_machine.state_machine import StateMachineGraph
from pse.state_machine.walker import Walker


class ArraySchemaAcceptor(ArrayAcceptor):
    def __init__(self, schema: Dict[str, Any], context):
        from pse.util.get_acceptor import (
            get_json_acceptor,
        )

        self.schema = schema
        self.context = context
        # Start of Selection
        graph: StateMachineGraph = {
            0: [
                (TextAcceptor("["), 1),
            ],
            1: [
                (WhitespaceAcceptor(), 2),
                (TextAcceptor("]"), "$"),
            ],
            2: [
                (get_json_acceptor(self.schema["items"], self.context), 3),
            ],
            3: [
                (WhitespaceAcceptor(), 4),
            ],
            4: [
                (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                (TextAcceptor("]"), "$"),
            ],
        }
        super().__init__(graph)

    def min_items(self) -> int:
        """
        Returns the minimum number of items in the array, according to the schema
        """
        return self.schema.get("minItems", 0)

    def max_items(self) -> int:
        """
        Returns the maximum number of items in the array, according to the schema
        """
        return self.schema.get("maxItems", 2**32)  # Arbitrary default

    @property
    def walker_class(self) -> Type[Walker]:
        return ArraySchemawalker


class ArraySchemawalker(ArrayWalker):
    """
    Walker for ArrayAcceptor
    """

    def __init__(self, acceptor: ArraySchemaAcceptor):
        super().__init__(acceptor)
        self.acceptor = acceptor

    def should_start_transition(self, transition_acceptor, target_state) -> bool:
        if self.current_state == 4 and target_state == 2:
            return len(self.value) < self.acceptor.max_items()
        if target_state == "$":
            return len(self.value) >= self.acceptor.min_items()
        return True
