from __future__ import annotations
import json
from typing import Dict, Any, Callable
from pse.acceptors.json.object_acceptor import ObjectAcceptor
from pse.util.errors import InvalidSchemaError
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.util.get_acceptor import get_json_acceptor


class ObjectSchemaAcceptor(ObjectAcceptor):

    def __init__(
        self,
        schema: Dict[str, Any],
        context: Dict[str, Any],
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
    ):
        self.schema = schema
        self.context = context
        self.properties: Dict[str, Any] = schema.get("properties", {})
        self.start_hook = start_hook
        self.end_hook = end_hook
        # Note that, according to the JSON schema specification, additional properties
        # should be allowed by default. The additionalProperties subschema can be used
        # to limit this. But we default to only allowing the properties defined in the
        # schema, because generally we don't want the LLM to generate at will. An
        # exception to this is when no properties are defined in the schema; in that
        # case we don't use this class but the superclass to allow any JSON object.
        if "additionalProperties" in schema:
            if schema["additionalProperties"] is False:
                self.allow_additional_properties = False
            else:
                # Implement handling for additionalProperties schema if necessary
                self.allow_additional_properties = True
        else:
            self.allow_additional_properties = True  # Default behavior per JSON Schema
        self.required_property_names = schema.get("required", [])
        for required_property_name in self.required_property_names:
            if required_property_name not in self.properties:
                raise InvalidSchemaError(
                    f"Required property '{required_property_name}' not defined"
                )

        assert self.properties is not None
        super().__init__()

    def get_edges(self, state):
        if state == 2:
            return [
                (
                    ObjectSchemaAcceptor.PropertyAcceptor(
                        prop_name,
                        prop_schema,
                        self.context,
                        self.start_hook,
                        self.end_hook,
                    ),
                    3,
                )
                for prop_name, prop_schema in self.properties.items()
            ]
        else:
            return super().get_edges(state)

    class Cursor(ObjectAcceptor.Cursor):
        """
        Cursor for ObjectAcceptor
        """

        def __init__(self, acceptor: ObjectSchemaAcceptor):
            super().__init__(acceptor)
            self.acceptor = acceptor

        def start_transition(
            self,
            transition_acceptor: ObjectSchemaAcceptor.PropertyAcceptor,
            target_state,
        ) -> bool:
            if target_state == "$":
                return all(
                    prop_name in self.value
                    for prop_name in self.acceptor.required_property_names
                )
            if self.current_state == 2 and target_state == 3:
                # Check if the property name is already in the object
                return transition_acceptor.prop_name not in self.value
            if self.current_state == 4 and target_state == 1:
                # Are all allowed properties already set?
                return len(self.value.keys()) < len(self.acceptor.properties)
            return True

    class PropertyAcceptor(ObjectAcceptor.PropertyAcceptor):
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

        class Cursor(ObjectAcceptor.PropertyAcceptor.Cursor):
            """
            Cursor for ObjectSchemaAcceptor.PropertyAcceptor
            """

            def __init__(self, acceptor: ObjectSchemaAcceptor.PropertyAcceptor):
                super().__init__(acceptor)
                self.acceptor = acceptor

            def complete_transition(
                self, transition_value, target_state, is_end_state
            ) -> bool:
                if not super().complete_transition(
                    transition_value, target_state, is_end_state
                ):
                    return False

                hooks: Dict[str, Callable] = (
                    self.acceptor.prop_schema.get("__hooks", {})
                )
                prop_name = self.acceptor.prop_name
                if target_state == 4:
                    if "value_start" in hooks:
                        hooks["value_start"](prop_name)
                elif is_end_state:
                    if "value_end" in hooks:
                        hooks["value_end"](prop_name, transition_value)
                return True

            def get_value(self):
                return (self.acceptor.prop_name, self.prop_value)
