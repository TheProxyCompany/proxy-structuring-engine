import pytest
from pse.acceptors.basic.boolean_acceptors import BooleanAcceptor
from pse.core.state_machine import StateMachine
from pse.core.walker import Walker
from typing import Iterable


# Fixture for BooleanAcceptor
@pytest.fixture
def boolean_acceptor():
    return BooleanAcceptor()


# Helper function to process input for BooleanAcceptor
def process_input(acceptor: StateMachine, token: str) -> Iterable[Walker]:
    walkers = acceptor.get_walkers()
    return [walker for _, walker in acceptor.advance_all(walkers, token)]


# Test for BooleanAcceptor
def test_accept_true(boolean_acceptor):
    accepted_walkers = process_input(boolean_acceptor, "true")
    assert any(
        walker.current_value is True for walker in accepted_walkers
    ), "Should have a walker with value True"


def test_accept_false(boolean_acceptor):
    accepted_walkers = process_input(boolean_acceptor, "false")
    assert any(
        walker.current_value is False for walker in accepted_walkers
    ), "Should have a walker with value False"


@pytest.mark.parametrize("token", ["tru", "fals", "True", "False", "TRUE", "FALSE"])
def test_reject_invalid_boolean(boolean_acceptor, token):
    accepted_walkers = list(process_input(boolean_acceptor, token))
    assert not any(
        walker.has_reached_accept_state() for walker in accepted_walkers
    ), f"Should not accept '{token}' as a valid boolean."


@pytest.mark.parametrize("token", [" true", "false ", " true ", "  false  "])
def test_accept_with_whitespace(boolean_acceptor, token):
    accepted_walkers = list(process_input(boolean_acceptor, token))
    assert not any(
        walker.has_reached_accept_state() for walker in accepted_walkers
    ), f"Should not accept '{token}' with whitespace."


@pytest.mark.parametrize("token", ["truex", "falsey", "true123", "false!"])
def test_extra_characters(boolean_acceptor, token):
    accepted_walkers = list(process_input(boolean_acceptor, token))
    assert not any(
        walker.has_reached_accept_state() for walker in accepted_walkers
    ), f"Should not accept '{token}' with extra characters."
