import unittest

from pse.state_machines.schema.enum_schema_acceptor import EnumSchemaAcceptor


class TestEnumSchemaAcceptor(unittest.TestCase):
    """
    Unit tests for the EnumSchemaAcceptor class to ensure full coverage of its functionalities.
    """

    def setUp(self) -> None:
        """
        Set up the default schema state_machine before each test.
        """
        self.default_schema: dict = {"enum": ["value1", "value2", "value3"]}
        self.state_machine: EnumSchemaAcceptor = EnumSchemaAcceptor(
            schema=self.default_schema
        )

    def test_accept_valid_enum_value(self) -> None:
        """
        Test that the state_machine correctly accepts a value present in the enum.
        """
        valid_value = "value1"
        walkers = self.state_machine.get_walkers()
        for char in valid_value:
            walkers = [
                walker for _, walker in EnumSchemaAcceptor.advance_all(walkers, char)
            ]
        for walker in walkers:
            self.assertTrue(walker.has_reached_accept_state())
            self.assertEqual(walker.get_current_value(), valid_value)

    def test_reject_invalid_enum_value(self) -> None:
        """
        Test that the state_machine correctly rejects a value not present in the enum.
        """
        invalid_value = "invalid_value"
        walkers = list(self.state_machine.get_walkers())
        for char in invalid_value:
            walkers = [
                walker for _, walker in EnumSchemaAcceptor.advance_all(walkers, char)
            ]
        # Final state should not be an accepted state
        self.assertFalse(
            any(walker.has_reached_accept_state() for walker in walkers),
            f"'{invalid_value}' should be rejected by the EnumSchemaAcceptor.",
        )

    def test_accept_multiple_enum_values(self) -> None:
        """
        Test that the state_machine correctly accepts multiple different valid enum values.
        """
        valid_values = ["value1", "value2", "value3"]
        for valid_value in valid_values:
            with self.subTest(valid_value=valid_value):
                walkers = self.state_machine.get_walkers()
                walkers = [
                    walker
                    for _, walker in EnumSchemaAcceptor.advance_all(
                        walkers, valid_value
                    )
                ]
                for walker in walkers:
                    self.assertTrue(walker.has_reached_accept_state())
                    self.assertEqual(walker.get_current_value(), valid_value)

    def test_partial_enum_value_rejection(self) -> None:
        """
        Test that the state_machine does not accept prefixes of valid enum values.
        """
        partial_value = "val"
        walkers = list(self.state_machine.get_walkers())
        for char in partial_value:
            walkers = [
                walker for _, walker in EnumSchemaAcceptor.advance_all(walkers, char)
            ]
        self.assertFalse(
            any(walker.has_reached_accept_state() for walker in walkers),
            f"Partial value '{partial_value}' should not be accepted by the EnumSchemaAcceptor.",
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
        state_machine: EnumSchemaAcceptor = EnumSchemaAcceptor(schema=schema)
        walkers = list(state_machine.get_walkers())
        self.assertFalse(
            any(walker.has_reached_accept_state() for walker in walkers),
            "Empty string should be rejected when enum is empty.",
        )

    def test_accept_enum_with_special_characters(self) -> None:
        """
        Test that the state_machine correctly handles enum values with special characters.
        """
        schema: dict = {"enum": ["val!@#", "val-123", "val_😊"]}
        state_machine: EnumSchemaAcceptor = EnumSchemaAcceptor(schema=schema)
        for special_value in schema["enum"]:
            with self.subTest(special_value=special_value):
                walkers = state_machine.get_walkers()
                walkers = [
                    walker
                    for _, walker in EnumSchemaAcceptor.advance_all(
                        walkers, special_value
                    )
                ]
                for walker in walkers:
                    self.assertTrue(walker.has_reached_accept_state())
                    self.assertEqual(
                        walker.get_current_value(), special_value.strip('"')
                    )
