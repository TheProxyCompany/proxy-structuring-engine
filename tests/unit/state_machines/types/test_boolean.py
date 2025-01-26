import pytest
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.state_machines.types.boolean import BooleanStateMachine


# Fixture for BooleanAcceptor
@pytest.fixture
def boolean_acceptor():
    return BooleanStateMachine()


# Helper function to process input for BooleanAcceptor
def process_input(state_machine: StateMachine, token: str) -> list[Stepper]:
    steppers = state_machine.get_steppers()
    return state_machine.advance_all_basic(steppers, token)


# Test for BooleanAcceptor
def test_accept_true() -> None:
    acceptor = BooleanStateMachine()
    steppers = acceptor.get_steppers()
    accepted_steppers = acceptor.advance_all_basic(steppers, "true")
    assert any(stepper.get_current_value() is True for stepper in accepted_steppers), (
        "Should have a stepper with value True"
    )


def test_accept_false(boolean_acceptor: BooleanStateMachine) -> None:
    steppers = boolean_acceptor.get_steppers()
    accepted_steppers = boolean_acceptor.advance_all_basic(steppers, "false")
    assert any(stepper.get_current_value() is False for stepper in accepted_steppers), (
        "Should have a stepper with value False"
    )


@pytest.mark.parametrize("token", ["tru", "fals", "True", "False", "TRUE", "FALSE"])
def test_reject_invalid_boolean(boolean_acceptor, token):
    accepted_steppers = list(process_input(boolean_acceptor, token))
    assert not any(
        stepper.has_reached_accept_state() for stepper in accepted_steppers
    ), f"Should not accept '{token}' as a valid boolean."


@pytest.mark.parametrize("token", [" true", "false ", " true ", "  false  "])
def test_accept_with_whitespace(boolean_acceptor, token):
    accepted_steppers = list(process_input(boolean_acceptor, token))
    assert not any(
        stepper.has_reached_accept_state() for stepper in accepted_steppers
    ), f"Should not accept '{token}' with whitespace."


@pytest.mark.parametrize("token", ["truex", "falsey", "true123", "false!"])
def test_extra_characters(boolean_acceptor, token):
    accepted_steppers = list(process_input(boolean_acceptor, token))
    assert not any(
        stepper.has_reached_accept_state() for stepper in accepted_steppers
    ), f"Should not accept '{token}' with extra characters."
