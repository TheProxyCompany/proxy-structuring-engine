import pytest
from pse_core.state_machine import StateMachine

from pse.types.base.loop import LoopStateMachine, LoopStepper
from pse.types.base.phrase import PhraseStateMachine
from pse.types.whitespace import WhitespaceStateMachine


@pytest.fixture
def basic_phrase_sm():
    """Fixture providing a basic PhraseStateMachine with 'test' as content."""
    return PhraseStateMachine("test")


@pytest.fixture
def comma_separator_sm():
    """Fixture providing a comma separator state machine."""
    return PhraseStateMachine(",")


@pytest.fixture
def whitespace_separator_sm():
    """Fixture providing a whitespace separator state machine."""
    return WhitespaceStateMachine()


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
    loop_sm = LoopStateMachine(
        inner_sm, min_loop_count=loop_count, max_loop_count=loop_count
    )

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


def test_loop_with_separator(comma_separator_sm):
    """Test loop with separator state machine."""
    inner_sm = PhraseStateMachine("item")
    loop_sm = LoopStateMachine(
        inner_sm,
        min_loop_count=2,
        max_loop_count=2,
        separator_state_machine=comma_separator_sm,
    )

    steppers = loop_sm.get_steppers()
    steppers = loop_sm.advance_all_basic(steppers, "item,item")

    assert len(steppers) == 1
    assert all(stepper.has_reached_accept_state() for stepper in steppers)

    assert all(stepper.get_current_value() == "item,item" for stepper in steppers)
    assert all(
        stepper.loop_count == 2
        for stepper in steppers
        if isinstance(stepper, LoopStepper)
    )


def test_loop_with_whitespace_separator(whitespace_separator_sm):
    """Test loop with whitespace separator state machine."""
    inner_sm = PhraseStateMachine("word")
    loop_sm = LoopStateMachine(
        inner_sm,
        min_loop_count=3,
        max_loop_count=3,
        separator_state_machine=whitespace_separator_sm,
    )

    steppers = loop_sm.get_steppers()
    steppers = loop_sm.advance_all_basic(steppers, "word word word")

    assert len(steppers) == 1
    assert all(stepper.has_reached_accept_state() for stepper in steppers)
    assert all(stepper.get_current_value() == "word word word" for stepper in steppers)
    assert all(
        stepper.loop_count == 3
        for stepper in steppers
        if isinstance(stepper, LoopStepper)
    )


@pytest.mark.parametrize(
    "separator,input_str,expected_value,expected_loop_count",
    [
        (PhraseStateMachine(","), "a,a,a", "a,a,a", 3),
        (PhraseStateMachine(";"), "a;a;a", "a;a;a", 3),
        (WhitespaceStateMachine(), "a a a", "a a a", 3),
        (PhraseStateMachine("-"), "a-a-a", "a-a-a", 3),
    ],
)
def test_multiple_separator_types(
    separator, input_str, expected_value, expected_loop_count
):
    """Test loop with different separator types."""
    inner_sm = PhraseStateMachine("a")
    loop_sm = LoopStateMachine(
        inner_sm, min_loop_count=3, max_loop_count=3, separator_state_machine=separator
    )

    steppers = loop_sm.get_steppers()
    steppers = loop_sm.advance_all_basic(steppers, input_str)

    assert len(steppers) == 1
    for stepper in steppers:
        assert isinstance(stepper, LoopStepper)
        assert stepper.get_current_value() == expected_value
        assert stepper.loop_count == expected_loop_count
        assert stepper.has_reached_accept_state()


def test_separator_history_exclusion(comma_separator_sm):
    """Test that separators are excluded from history."""
    inner_sm = PhraseStateMachine("item")
    loop_sm = LoopStateMachine(
        inner_sm,
        min_loop_count=2,
        max_loop_count=3,
        separator_state_machine=comma_separator_sm,
        track_separator=False,
    )

    steppers = loop_sm.get_steppers()
    steppers = loop_sm.advance_all_basic(steppers, "item,item")

    # Check the loop stepper's history - should only contain item steppers, not separators
    stepper = steppers[0]
    assert len(stepper.history) == 2

    # Verify each history item is from the inner state machine, not the separator
    for history_stepper in stepper.history:
        assert history_stepper.state_machine == inner_sm
        assert history_stepper.get_current_value() == "item"


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


def test_complex_inner_with_separator(comma_separator_sm):
    """Test loop with complex inner state machine and separator."""
    inner_sm = StateMachine(
        {
            0: [(PhraseStateMachine("["), 1)],
            1: [(PhraseStateMachine("]"), 2)],
        },
        start_state=0,
        end_states=[2],
    )

    loop_sm = LoopStateMachine(
        inner_sm,
        min_loop_count=2,
        max_loop_count=2,
        separator_state_machine=comma_separator_sm,
    )

    steppers = loop_sm.get_steppers()
    steppers = loop_sm.advance_all_basic(steppers, "[],[")
    assert len(steppers) == 1
    steppers = loop_sm.advance_all_basic(steppers, "]")

    assert len(steppers) == 1
    assert all(stepper.has_reached_accept_state() for stepper in steppers)
    assert all(stepper.get_current_value() == "[],[]" for stepper in steppers)


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


def test_nested_loop_with_separators(comma_separator_sm, whitespace_separator_sm):
    """Test nested loops with different separators."""
    # Inner loop with comma separator
    inner_sm = PhraseStateMachine("item")
    inner_loop_sm = LoopStateMachine(
        inner_sm,
        min_loop_count=2,
        max_loop_count=2,
        separator_state_machine=comma_separator_sm,
    )

    # Outer loop with whitespace separator
    outer_loop_sm = LoopStateMachine(
        inner_loop_sm,
        min_loop_count=2,
        max_loop_count=2,
        separator_state_machine=whitespace_separator_sm,
    )

    steppers = outer_loop_sm.get_steppers()
    steppers = outer_loop_sm.advance_all_basic(steppers, "item,item item,item")

    assert len(steppers) == 1
    assert all(stepper.has_reached_accept_state() for stepper in steppers)
    assert all(
        stepper.get_current_value() == "item,item item,item" for stepper in steppers
    )


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


def test_loop_max_count_with_separator(comma_separator_sm):
    """Test loop max count with separator."""
    inner_sm = PhraseStateMachine("item")
    loop_sm = LoopStateMachine(
        inner_sm,
        min_loop_count=1,
        max_loop_count=2,
        separator_state_machine=comma_separator_sm,
    )

    steppers = loop_sm.get_steppers()
    # First item
    steppers = loop_sm.advance_all_basic(steppers, "item")
    assert all(
        stepper.loop_count == 1
        for stepper in steppers
        if isinstance(stepper, LoopStepper)
    )

    # Separator
    steppers = loop_sm.advance_all_basic(steppers, ",")
    assert all(
        stepper.loop_count == 1
        for stepper in steppers
        if isinstance(stepper, LoopStepper)
    )  # Count shouldn't change

    # Second item
    steppers = loop_sm.advance_all_basic(steppers, "item")
    assert all(
        stepper.loop_count == 2
        for stepper in steppers
        if isinstance(stepper, LoopStepper)
    )

    # Should not accept more input due to max_loop_count
    assert all(not stepper.should_start_step(",") for stepper in steppers)


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

    # Test with no separator
    final_states = stepper.get_final_state()
    assert final_states == stepper.history
