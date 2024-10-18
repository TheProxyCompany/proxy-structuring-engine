from __future__ import annotations
from typing import Any, Dict
from pse.acceptors.collections.array_acceptor import ArrayAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.state_machine.state_machine import StateMachineGraph


class ArraySchemaAcceptor(ArrayAcceptor):
    """
    TODO
    {
      "type": "array",
      "items": { "type": "number" },
      "uniqueItems": true
    }
    {
      "type": "array",
      "prefixItems": [
        { "type": "number" },
        { "type": "string" },
        { "enum": ["Street", "Avenue", "Boulevard"] },
        { "enum": ["NW", "NE", "SW", "SE"] }
      ],
      "items": { "type": "string" } # Constrain additional items to type string
      "items": false # Do not allow items beyond the prefixItem
      "unevaluatedItems": false # All prefixItems are required
      "unevaluatedItems": { "const": "N/A" } # Default value for prefixItems
    }
    {
      "type": "array",
      "contains": {
        "type": "number" # Contains at least one number
      },
      "minContains": 2, # Must contain at least two numbers
      "maxContains": 3 # Must contain at most three numbers
    }
    """

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

    class Cursor(ArrayAcceptor.Cursor):
        """
        Cursor for ArrayAcceptor
        """

        def __init__(self, acceptor: ArraySchemaAcceptor):
            super().__init__(acceptor)
            self.acceptor = acceptor

        def start_transition(self, transition_acceptor, target_state) -> bool:
            if self.current_state == 4 and target_state == 2:
                return len(self.value) < self.acceptor.max_items()
            if target_state == "$":
                return len(self.value) >= self.acceptor.min_items()
            return True
