from __future__ import annotations

from typing import Any

from pse_core import State

from pse.state_machines.basic.text_acceptor import TextAcceptor
from pse.state_machines.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.state_machines.collections.array_acceptor import ArrayAcceptor, ArrayWalker
from pse.state_machines.collections.sequence_acceptor import SequenceAcceptor


class ArraySchemaAcceptor(ArrayAcceptor):
    def __init__(self, schema: dict[str, Any], context: dict[str, Any]) -> None:
        from pse.state_machines.get_state_machine import (
            get_state_machine,
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
                    (get_state_machine(self.schema["items"], self.context), 3),
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

    def get_new_walker(self, state: State | None = None) -> ArraySchemaWalker:
        return ArraySchemaWalker(self, state)

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

    def __init__(
        self, state_machine: ArraySchemaAcceptor, current_state: State | None = None
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: ArraySchemaAcceptor = state_machine

    def should_start_transition(self, token: str) -> bool:
        if (self.current_state == 2 and self.target_state == 3) or (
            self.current_state == 4 and self.target_state == 2
        ):
            return len(self.value) < self.state_machine.max_items()
        if self.target_state == "$":
            return len(self.value) >= self.state_machine.min_items()

        return super().should_start_transition(token)
