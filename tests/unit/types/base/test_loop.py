import pytest
from pse_core.state_machine import StateMachine

from pse.types.base.loop import LoopStateMachine, LoopStepper
from pse.types.base.phrase import PhraseStateMachine


@pytest.fixture
def basic_phrase_sm():
    """Fixture providing a basic PhraseStateMachine with 'test' as content."""
    return PhraseStateMachine("test")


@pytest.fixture
def basic_loop_sm(basic_phrase_sm):
    """Fixture providing a LoopStateMachine with loop_count=1."""
    return LoopStateMachine(basic_phrase_sm, min_loop_count=1, max_loop_count=1)


def test_basic_loop(basic_loop_sm: LoopStateMachine):
    """Test basic loop functionality with a single iteration."""
    steppers = basic_loop_sm.get_steppers()
    assert len(steppers) == 1
    assert isinstance(steppers[0], LoopStepper)

    steppers = basic_loop_sm.advance_all_basic(steppers, "test")
    assert len(steppers) == 1
    assert all(stepper.has_reached_accept_state() for stepper in steppers)
    assert all(stepper.get_current_value() == "test" for stepper in steppers)


@pytest.mark.parametrize(
    "loop_count,input_str,expected_value,expected_loop_count",
    [
        (3, "aaa", "aaa", 3),  # Full completion
        (2, "aa", "aa", 2),  # Exact match
        (4, "aa", "aa", 2),  # Incomplete
    ],
)
def test_multiple_loops(loop_count, input_str, expected_value, expected_loop_count):
    """Test loop with multiple iterations using different parameters."""
    inner_sm = PhraseStateMachine("a")
    loop_sm = LoopStateMachine(inner_sm, min_loop_count=loop_count, max_loop_count=loop_count)

    steppers = loop_sm.get_steppers()
    steppers = loop_sm.advance_all_basic(steppers, input_str)

    assert len(steppers) == 1
    for stepper in steppers:
        assert isinstance(stepper, LoopStepper)
        assert stepper.get_current_value() == expected_value
        assert stepper.loop_count == expected_loop_count
        if expected_loop_count == loop_count:
            assert stepper.has_reached_accept_state()
        else:
            assert not stepper.has_reached_accept_state()

def test_loop_with_complex_inner_state():
    """Test loop with a more complex inner state machine."""
    inner_sm = StateMachine(
        {
            0: [(PhraseStateMachine("("), 1)],
            1: [(PhraseStateMachine(")"), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    loop_sm = LoopStateMachine(inner_sm, min_loop_count=2, max_loop_count=2)

    steppers = loop_sm.get_steppers()
    steppers = loop_sm.advance_all_basic(steppers, "()()")

    assert len(steppers) == 1
    assert all(stepper.has_reached_accept_state() for stepper in steppers)
    assert all(stepper.get_current_value() == "()()" for stepper in steppers)

def test_nested_loop():
    """Test nested loop functionality."""
    inner_sm = PhraseStateMachine("a")
    loop_sm = LoopStateMachine(inner_sm, min_loop_count=1, max_loop_count=2)
    sm = StateMachine(
        {
            0: [(loop_sm, 1)],
            1: [(PhraseStateMachine("b"), 2)],
        },
        start_state=0,
        end_states=[2],
    )
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "aa")
    assert len(steppers) == 1
    steppers = sm.advance_all_basic(steppers, "b")
    assert all(stepper.has_reached_accept_state() for stepper in steppers)
    assert all(stepper.get_current_value() == "aab" for stepper in steppers)


def test_loop_max_count():
    """
    Test the loop max_count logic directly.

    This test focuses on the should_start_step method to test the part
    that checks if loop_count >= max_loop_count. We'll create a stepper
    and manually set its loop_count.
    """
    inner_sm = PhraseStateMachine("a")
    loop_sm = LoopStateMachine(inner_sm, min_loop_count=1, max_loop_count=3)

    # Create stepper directly
    steppers = loop_sm.get_steppers()
    stepper = steppers[0]
    assert isinstance(stepper, LoopStepper)

    # Initial state, loop_count = 0
    assert stepper.should_start_step("a")

    # Set loop_count to max_loop_count
    stepper.loop_count = 3

    # Now should_start_step should be False due to max_loop_count check
    assert not stepper.should_start_step("a")

def test_can_accept_more_input_max_loop_count():
    """Test can_accept_more_input with max_loop_count."""
    inner_sm = PhraseStateMachine("a")
    loop_sm = LoopStateMachine(inner_sm, min_loop_count=1, max_loop_count=3)
    
    # Create stepper
    steppers = loop_sm.get_steppers()
    stepper = steppers[0]
    
    # Initially can accept more
    assert stepper.can_accept_more_input() is True
    
    # Set loop_count to max_loop_count
    stepper.loop_count = 3
    assert stepper.can_accept_more_input() is False
    
    # Reset to test with negative max_loop_count (unlimited)
    loop_sm = LoopStateMachine(inner_sm, min_loop_count=1, max_loop_count=-1)
    steppers = loop_sm.get_steppers()
    stepper = steppers[0]
    stepper.loop_count = 100  # Even with high count
    assert stepper.can_accept_more_input() is True

def test_get_final_state():
    """Test get_final_state in LoopStepper."""
    # Set up a mock Stepper
    inner_sm = PhraseStateMachine("a")
    loop_sm = LoopStateMachine(inner_sm, min_loop_count=1)
    stepper = loop_sm.get_new_stepper()
    
    # Set internal loop stepper state for testing
    sub_stepper = inner_sm.get_new_stepper()
    stepper.sub_stepper = sub_stepper
    stepper.history = [sub_stepper]
    
    # Test with no whitespace separators
    final_states = stepper.get_final_state()
    assert final_states == stepper.history

def test_has_reached_accept_state_with_min_loop_count():
    """Test has_reached_accept_state method with different min_loop_count values."""
    # Loop with min_loop_count=2
    inner_sm = PhraseStateMachine("a")
    loop_sm = LoopStateMachine(inner_sm, min_loop_count=2)
    stepper = loop_sm.get_new_stepper()
    
    # Initially should not be in accept state
    assert not stepper.has_reached_accept_state()
    
    # Set loop_count to 1 (less than min)
    stepper.loop_count = 1
    assert not stepper.has_reached_accept_state()
    
    # Set loop_count to 2 (equal to min)
    stepper.loop_count = 2
    assert stepper.has_reached_accept_state()
