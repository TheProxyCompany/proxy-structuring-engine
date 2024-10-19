# import pytest
# from unittest.mock import MagicMock
# from typing import Any, Dict

# from pse.schema_acceptors.object_schema_acceptor import ObjectSchemaAcceptor
# from pse.util.errors import InvalidSchemaError


# @pytest.fixture
# def base_context() -> Dict[str, Any]:
#     """Fixture to initialize the common test variables before each test."""
#     return {"defs": {}, "path": "root"}


# def test_initialization_valid_schema(base_context: Dict[str, Any]) -> None:
#     """
#     Test initializing ObjectSchemaAcceptor with a valid schema.
#     Ensures that the instance is created without errors.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "name": {"type": "string"},
#             "age": {"type": "number"},
#         },
#         "required": ["name"],
#     }
#     acceptor = ObjectSchemaAcceptor(schema, base_context)
#     assert acceptor.schema == schema, "Schema should be set correctly."
#     assert acceptor.context == base_context, "Context should be set correctly."
#     assert (
#         acceptor.properties == schema["properties"]
#     ), "Properties should be extracted correctly."
#     assert acceptor.required_property_names == [
#         "name"
#     ], "Required properties should be set correctly."


# def test_initialization_missing_required_property_in_properties(
#     base_context: Dict[str, Any]
# ) -> None:
#     """
#     Test initializing ObjectSchemaAcceptor where a required property is not defined in properties.
#     Ensures that InvalidSchemaError is raised.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "age": {"type": "number"},
#         },
#         "required": ["name"],
#     }
#     with pytest.raises(InvalidSchemaError):
#         ObjectSchemaAcceptor(schema, base_context)


# def test_get_edges_state_3(base_context: Dict[str, Any]) -> None:
#     """
#     Test the get_edges method for state 3.
#     Ensures that property acceptors are returned correctly.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "name": {"type": "string"},
#             "age": {"type": "number"},
#         },
#         "required": ["name"],
#     }
#     acceptor = ObjectSchemaAcceptor(schema, base_context)
#     edges = acceptor.get_edges(2)
#     assert len(edges) == 2, "Should return one edge per property."
#     for edge, state in edges:
#         assert isinstance(
#             edge, ObjectSchemaAcceptor.PropertyAcceptor
#         ), "Edge should be a PropertyAcceptor instance."
#         assert state == 3, "State should transition to 3."


# def test_get_edges_other_states(base_context: Dict[str, Any]) -> None:
#     """
#     Test the get_edges method for states other than 2.
#     Ensures that the superclass method is called.
#     """
#     schema = {
#         "type": "object",
#         "properties": {"name": {"type": "string"}},
#         "required": ["name"],
#     }
#     acceptor = ObjectSchemaAcceptor(schema, base_context)
#     edges = acceptor.get_edges(4)  # State not specifically handled
#     assert isinstance(edges, list), "Edges should be a list."
#     # Further assertions based on superclass behavior could be added here


# def test_property_acceptor_complete_transition_with_hooks(
#     base_context: Dict[str, Any]
# ) -> None:
#     """
#     Test the complete_transition method of PropertyAcceptor with hooks.
#     Ensures that hooks are called appropriately during state transitions.
#     """
#     # Flags to track hook invocations
#     flags = {
#         "value_start_called": False,
#         "value_end_called": False,
#         "value_start_prop": "",
#         "value_end_prop": "",
#         "value_end_val": None,
#     }

#     def mock_value_start(prop_name: str) -> None:
#         """Mock function to simulate value_start hook."""
#         flags["value_start_called"] = True
#         flags["value_start_prop"] = prop_name

#     def mock_value_end(prop_name: str, value: Any) -> None:
#         """Mock function to simulate value_end hook."""
#         flags["value_end_called"] = True
#         flags["value_end_prop"] = prop_name
#         flags["value_end_val"] = value

#     schema = {
#         "type": "object",
#         "properties": {
#             "username": {
#                 "type": "string",
#                 "__hooks": {
#                     "value_start": mock_value_start,
#                     "value_end": mock_value_end,
#                 },
#             }
#         },
#         "required": ["username"],
#     }
#     acceptor = ObjectSchemaAcceptor(schema, base_context)
#     property_acceptor = acceptor.get_edges(2)[0][0]

#     assert isinstance(
#         property_acceptor, ObjectSchemaAcceptor.PropertyAcceptor
#     ), "Should be an instance of PropertyAcceptor."
#     cursor = property_acceptor.Cursor(property_acceptor)

#     # Complete transition to start value
#     result_start = cursor.complete_transition("JohnDoe", 4, False)
#     assert result_start, "Transition should return True."
#     assert flags["value_start_called"], "value_start hook should be called."
#     assert (
#         flags["value_start_prop"] == "username"
#     ), "value_start should receive correct property name."

#     # Complete transition to end value
#     result_end = cursor.complete_transition("JohnDoe", 5, True)
#     assert result_end, "Transition should return True."
#     assert flags["value_end_called"], "value_end hook should be called."
#     assert (
#         flags["value_end_prop"] == "username"
#     ), "value_end should receive correct property name."
#     assert (
#         flags["value_end_val"] == "JohnDoe"
#     ), "value_end should receive correct value."


# def test_cursor_initialization(base_context: Dict[str, Any]) -> None:
#     """
#     Test the initialization of ObjectSchemaAcceptor.Cursor.
#     Ensures that the cursor starts with an empty dictionary and correct acceptor reference.
#     """
#     schema = {"type": "object", "properties": {}, "required": []}
#     acceptor = ObjectSchemaAcceptor(schema, base_context)
#     cursor = acceptor.Cursor(acceptor)
#     assert cursor.value == {}, "Cursor should initialize with an empty dictionary."
#     assert (
#         cursor.acceptor is acceptor
#     ), "Cursor's acceptor should reference the ObjectSchemaAcceptor instance."


# def test_cursor_start_transition_valid(base_context: Dict[str, Any]) -> None:
#     """
#     Test the start_transition method of ObjectSchemaAcceptor.Cursor with valid transitions.
#     Ensures that transitions return True when conditions are met.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "id": {"type": "number"},
#             "email": {"type": "string"},
#         },
#         "required": ["id"],
#     }
#     acceptor = ObjectSchemaAcceptor(schema, base_context)
#     cursor = acceptor.Cursor(acceptor)
#     cursor.value = {"id": 123}

#     # Transition to add 'email'
#     property_acceptor = ObjectSchemaAcceptor.PropertyAcceptor(
#         "email", schema["properties"]["email"], base_context
#     )
#     result = cursor.start_transition(property_acceptor, 4)
#     assert result, "Transition should return True when adding a new property."

#     # Transition to end state
#     result_end = cursor.start_transition(property_acceptor, "$")
#     assert (
#         result_end
#     ), "Transition to end state should return True when required properties are present."


# def test_cursor_start_transition_invalid(base_context: Dict[str, Any]) -> None:
#     """
#     Test the start_transition method of ObjectSchemaAcceptor.Cursor with invalid transitions.
#     Ensures that transitions return False when conditions are not met.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "id": {"type": "number"},
#             "email": {"type": "string"},
#         },
#         "required": ["id"],
#     }
#     acceptor = ObjectSchemaAcceptor(schema, base_context)
#     cursor = acceptor.Cursor(acceptor)
#     cursor.value = {"id": 123, "email": "user@example.com"}

#     # Attempt to add 'email' again
#     property_acceptor = ObjectSchemaAcceptor.PropertyAcceptor(
#         "email", schema["properties"]["email"], base_context
#     )
#     cursor.current_state = 2
#     cursor.target_state = 3
#     result = cursor.start_transition(property_acceptor, 3)
#     assert (
#         not result
#     ), "Transition should return False when adding an existing property."

#     # Attempt to transition to end state without required properties
#     cursor.value = {}  # Remove required property
#     result_end = cursor.start_transition(property_acceptor, "$")
#     assert (
#         not result_end
#     ), "Transition to end state should return False when required properties are missing."


# def test_value_started_hook_not_string(base_context: Dict[str, Any]) -> None:
#     """
#     Test that the value_started_hook is not called prematurely.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "id": {"type": "number"},
#             "email": {"type": "string"},
#         },
#         "required": ["id"],
#     }

#     started_hook = MagicMock()
#     acceptor = ObjectSchemaAcceptor(schema, base_context, start_hook=started_hook)
#     cursors = list(acceptor.get_cursors())
#     for char in '{"id"':
#         cursors = list(acceptor.advance_all(cursors, char))
#     # The hook should not be called yet
#     started_hook.assert_not_called()

#     # Continue parsing
#     for char in '": 123}':
#         cursors = list(acceptor.advance_all(cursors, char))

#     assert any(
#         cursor.in_accepted_state() for cursor in cursors
#     ), "Transition to end state should return True when required properties are present."
#     started_hook.assert_not_called()


# def test_value_started_hook(base_context: Dict[str, Any]) -> None:
#     """
#     Test that the value_started_hook is called with the correct property name when a property's value starts.
#     """
#     schema = {
#         "type": "object",
#         "properties": {"id": {"type": "number"}, "email": {"type": "string"}},
#         "required": ["id"],
#     }

#     started_hook = MagicMock()
#     acceptor = ObjectSchemaAcceptor(schema, base_context, start_hook=started_hook)
#     cursors = list(acceptor.get_cursors())
#     for char in '{"email": ':
#         cursors = list(acceptor.advance_all(cursors, char))

#     started_hook.assert_not_called()
#     # Simulate starting a string value
#     cursors = list(acceptor.advance_all(cursors, '"'))
#     # The hook should be called now
#     started_hook.assert_called_once()


# def test_value_ended_hook(base_context: Dict[str, Any]) -> None:
#     """
#     Test that the value_ended_hook is called with the correct property name and value when a property's value ends.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "id": {"type": "number"},
#             "email": {"type": "string"},
#         },
#         "required": ["email"],
#     }
#     ended_hook = MagicMock()
#     acceptor = ObjectSchemaAcceptor(schema, base_context, end_hook=ended_hook)
#     cursors = list(acceptor.get_cursors())

#     additional_chars = '{"id": 1, "email":"user'
#     for char in additional_chars:
#         cursors = list(acceptor.advance_all(cursors, char))
#     ended_hook.assert_not_called()

#     # Finish the string and the object
#     for char in '"}':
#         cursors = list(acceptor.advance_all(cursors, char))

#     assert any(
#         cursor.in_accepted_state() for cursor in cursors
#     ), "Transition to end state should return True when all properties are present."
#     ended_hook.assert_called_once()


# def test_complex_json_structure(base_context: Dict[str, Any]) -> None:
#     """Test parsing a complex JSON structure."""
#     schema = {
#         "type": "object",
#         "properties": {
#             "name": {"type": "const", "const": "metacognition"},
#             "arguments": {
#                 "type": "object",
#                 "properties": {
#                     "chain_of_thought": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                     },
#                     "feelings": {
#                         "type": ["string"],
#                         "nullable": True,
#                         "default": None,
#                     },
#                 },
#                 "required": ["chain_of_thought"],
#             },
#         },
#         "required": ["name", "arguments"],
#     }
#     acceptor = ObjectSchemaAcceptor(schema, base_context)

#     # Test initialization
#     assert acceptor.schema == schema, "Schema should be set correctly."
#     assert acceptor.context == base_context, "Context should be set correctly."
#     assert (
#         acceptor.properties == schema["properties"]
#     ), "Properties should be extracted correctly."
#     assert acceptor.required_property_names == [
#         "name",
#         "arguments",
#     ], "Required properties should be set correctly."

#     # Test parsing valid JSON input
#     cursors = acceptor.get_cursors()
#     valid_json = '{"name": "metacognition", "arguments": {"chain_of_thought": ["Thought 1", "Thought 2"]}}'
#     for char in valid_json:
#         cursors = acceptor.advance_all(cursors, char)

#     assert any(
#         cursor.in_accepted_state() for cursor in cursors
#     ), "Transition to end state should return True for valid input."
