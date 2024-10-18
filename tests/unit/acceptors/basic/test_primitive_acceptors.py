import pytest
from pse.acceptors.basic.primitive_acceptors import (
    BooleanAcceptor,
    NullAcceptor,
)
from pse.state_machine.cursor import Cursor
from typing import Iterable

# Fixture for BooleanAcceptor
@pytest.fixture
def boolean_acceptor():
    return BooleanAcceptor()

# Fixture for NullAcceptor
@pytest.fixture
def null_acceptor():
    return NullAcceptor()

# Helper function to process input for BooleanAcceptor
def process_input(acceptor, input_str: str) -> Iterable[Cursor]:
    cursors = acceptor.get_cursors()
    return acceptor.advance_all(cursors, input_str)

# Test for BooleanAcceptor
def test_accept_true(boolean_acceptor):
    accepted_cursors = process_input(boolean_acceptor, "true")
    assert any(cursor.get_value() is True for cursor in accepted_cursors), "Should have a cursor with value True"

def test_accept_false(boolean_acceptor):
    accepted_cursors = process_input(boolean_acceptor, "false")
    assert any(cursor.get_value() is False for cursor in accepted_cursors), "Should have a cursor with value False"

@pytest.mark.parametrize("input_str", ["tru", "fals", "True", "False", "TRUE", "FALSE"])
def test_reject_invalid_boolean(boolean_acceptor, input_str):
    accepted_cursors = list(process_input(boolean_acceptor, input_str))
    assert not any(cursor.in_accepted_state() for cursor in accepted_cursors), f"Should not accept '{input_str}' as a valid boolean."

@pytest.mark.parametrize("input_str", [" true", "false ", " true ", "  false  "])
def test_accept_with_whitespace(boolean_acceptor, input_str):
    accepted_cursors = list(process_input(boolean_acceptor, input_str))
    assert not any(cursor.in_accepted_state() for cursor in accepted_cursors), f"Should not accept '{input_str}' with whitespace."

@pytest.mark.parametrize("input_str", ["truex", "falsey", "true123", "false!"])
def test_extra_characters(boolean_acceptor, input_str):
    accepted_cursors = list(process_input(boolean_acceptor, input_str))
    assert not any(cursor.in_accepted_state() for cursor in accepted_cursors), f"Should not accept '{input_str}' with extra characters."

# Test for NullAcceptor
def test_accept_null(null_acceptor):
    accepted_cursors = process_input(null_acceptor, "null")
    for cursor in accepted_cursors:
        assert cursor.get_value() == "null", "Should have a cursor with value 'null'"
        return
    pytest.fail("Should have a cursor with value 'null'")

@pytest.mark.parametrize("input_str", ["nul", "Null", "NULL", "nulll"])
def test_reject_invalid_null(null_acceptor, input_str):
    accepted_cursors = list(process_input(null_acceptor, input_str))
    assert not any(cursor.in_accepted_state() for cursor in accepted_cursors), f"Should not accept '{input_str}' as a valid null value."

@pytest.mark.parametrize("input_str", [" null", "null ", " null ", "  null  "])
def test_accept_with_whitespace_null(null_acceptor, input_str):
    accepted_cursors = list(process_input(null_acceptor, input_str))
    assert not any(cursor.in_accepted_state() for cursor in accepted_cursors), f"Should not accept '{input_str}' with whitespace."

@pytest.mark.parametrize("input_str", ["nullx", "null123", "null!", "nullnull"])
def test_extra_characters_null(null_acceptor, input_str):
    accepted_cursors = list(process_input(null_acceptor, input_str))
    assert not any(cursor.in_accepted_state() for cursor in accepted_cursors), f"Should not accept '{input_str}' with extra characters."
