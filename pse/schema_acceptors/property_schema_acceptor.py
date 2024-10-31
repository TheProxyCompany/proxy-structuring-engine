from pse.acceptors.json.property_acceptor import PropertyAcceptor, Propertywalker
from pse.state_machine.state_machine import Walker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from typing import Dict, Any, Callable, Type
import json

from pse.util.get_acceptor import get_json_acceptor


class PropertySchemaAcceptor(PropertyAcceptor):
    """
    Acceptor for an object property according to the schema.

    Args:
        prop_name (str): The name of the property.
        prop_schema (Dict[str, Any]): The schema of the property.
        context (Dict[str, Any]): The parsing context.
        value_started_hook (Callable | None, optional): Hook called when value parsing starts.
        value_ended_hook (Callable | None, optional): Hook called when value parsing ends.
    """

    def __init__(
        self,
        prop_name: str,
        prop_schema: Dict[str, Any],
        context: Dict[str, Any],
        value_started_hook: Callable | None = None,
        value_ended_hook: Callable | None = None,
    ):
        self.prop_name = prop_name
        self.prop_schema = prop_schema
        self.prop_context = {
            "defs": context.get("defs", {}),
            "path": f"{context.get('path', '')}/{prop_name}",
        }
        super().__init__(
            [
                TextAcceptor(json.dumps(self.prop_name)),
                WhitespaceAcceptor(),
                TextAcceptor(":"),
                WhitespaceAcceptor(),
                get_json_acceptor(
                    self.prop_schema,
                    self.prop_context,
                    value_started_hook,
                    value_ended_hook,
                ),
            ]
        )

    @property
    def walker_class(self) -> Type[Walker]:
        return PropertySchemawalker


class PropertySchemawalker(Propertywalker):
    """
    Walker for PropertySchemaAcceptor
    """

    def __init__(self, acceptor: PropertySchemaAcceptor):
        super().__init__(acceptor)
        self.acceptor = acceptor

    def should_complete_transition(
        self, transition_value, target_state, is_end_state
    ) -> bool:
        if not super().should_complete_transition(
            transition_value, target_state, is_end_state
        ):
            return False

        hooks: Dict[str, Callable] = self.acceptor.prop_schema.get("__hooks", {})
        prop_name = self.acceptor.prop_name
        if target_state == 4:
            if "value_start" in hooks:
                hooks["value_start"](prop_name)
        elif is_end_state:
            if "value_end" in hooks:
                hooks["value_end"](prop_name, transition_value)
        return True

    def accumulated_value(self):
        return (self.acceptor.prop_name, self.prop_value)
