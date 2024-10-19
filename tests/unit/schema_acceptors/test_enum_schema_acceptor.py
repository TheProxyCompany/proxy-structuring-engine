import unittest
from pse.schema_acceptors.enum_schema_acceptor import EnumSchemaAcceptor

class TestEnumSchemaAcceptor(unittest.TestCase):
    """
    Unit tests for the EnumSchemaAcceptor class to ensure full coverage of its functionalities.
    """

    def setUp(self) -> None:
        """
        Set up the default schema acceptor before each test.
        """
        self.default_schema: dict = {"enum": ["value1", "value2", "value3"]}
        self.acceptor: EnumSchemaAcceptor = EnumSchemaAcceptor(schema=self.default_schema)

    def test_accept_valid_enum_value(self) -> None:
        """
        Test that the acceptor correctly accepts a value present in the enum.
        """
        valid_value = "value1"
        cursors = self.acceptor.get_cursors()
        for char in valid_value:
            cursors = EnumSchemaAcceptor.advance_all(cursors, char)
        for cursor in cursors:
            self.assertTrue(cursor.in_accepted_state())
            self.assertEqual(cursor.get_value(), valid_value)

    def test_reject_invalid_enum_value(self) -> None:
        """
        Test that the acceptor correctly rejects a value not present in the enum.
        """
        invalid_value = "invalid_value"
        cursors = list(self.acceptor.get_cursors())
        for char in invalid_value:
            cursors = EnumSchemaAcceptor.advance_all(cursors, char)
        # Final state should not be an accepted state
        self.assertFalse(
            any(cursor.in_accepted_state() for cursor in cursors),
            f"'{invalid_value}' should be rejected by the EnumSchemaAcceptor."
        )

    def test_accept_multiple_enum_values(self) -> None:
        """
        Test that the acceptor correctly accepts multiple different valid enum values.
        """
        valid_values = ["value1", "value2", "value3"]
        for valid_value in valid_values:
            with self.subTest(valid_value=valid_value):
                cursors = self.acceptor.get_cursors()
                cursors = EnumSchemaAcceptor.advance_all(cursors, valid_value)
                for cursor in cursors:
                    self.assertTrue(cursor.in_accepted_state())
                    self.assertEqual(cursor.get_value(), valid_value)

    def test_partial_enum_value_rejection(self) -> None:
        """
        Test that the acceptor does not accept prefixes of valid enum values.
        """
        partial_value = "val"
        cursors = list(self.acceptor.get_cursors())
        for char in partial_value:
            cursors = EnumSchemaAcceptor.advance_all(cursors, char)
        self.assertFalse(
            any(cursor.in_accepted_state() for cursor in cursors),
            f"Partial value '{partial_value}' should not be accepted by the EnumSchemaAcceptor."
        )

    def test_init_without_enum_key(self) -> None:
        """
        Test initializing EnumSchemaAcceptor without an 'enum' key in the schema.
        Ensures that it raises a KeyError.
        """
        schema: dict = {}
        with self.assertRaises(KeyError):
            EnumSchemaAcceptor(schema=schema)

    def test_init_with_non_list_enum(self) -> None:
        """
        Test initializing EnumSchemaAcceptor with a non-list 'enum' value.
        Ensures that it raises a TypeError.
        """
        schema: dict = {"enum": "not_a_list"}
        with self.assertRaises(TypeError):
            EnumSchemaAcceptor(schema=schema)

    def test_accept_empty_string_when_enum_is_empty(self) -> None:
        """
        Test that accepting an empty string behaves correctly when the enum is empty.
        """
        schema: dict = {"enum": []}
        acceptor: EnumSchemaAcceptor = EnumSchemaAcceptor(schema=schema)
        cursors = list(acceptor.get_cursors())
        self.assertFalse(
            any(cursor.in_accepted_state() for cursor in cursors),
            "Empty string should be rejected when enum is empty."
        )

    def test_accept_enum_with_special_characters(self) -> None:
        """
        Test that the acceptor correctly handles enum values with special characters.
        """
        schema: dict = {"enum": ["val!@#", "val-123", "val_😊"]}
        acceptor: EnumSchemaAcceptor = EnumSchemaAcceptor(schema=schema)
        for special_value in schema["enum"]:
            with self.subTest(special_value=special_value):
                cursors = acceptor.get_cursors()
                cursors = EnumSchemaAcceptor.advance_all(cursors, special_value)
                for cursor in cursors:
                    self.assertTrue(cursor.in_accepted_state())
                    self.assertEqual(cursor.get_value(), special_value.strip('"'))