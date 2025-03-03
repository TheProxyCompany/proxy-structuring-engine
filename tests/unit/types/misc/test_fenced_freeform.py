from pse.types.misc.fenced_freeform import FencedFreeformStateMachine


def test_fenced_freeform_default_delimiters():
    """Test FencedFreeformStateMachine with default delimiters."""
    sm = FencedFreeformStateMachine()
    input_sequence = "```\nSome freeform text\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_fenced_freeform_custom_delimiter():
    """Test FencedFreeformStateMachine with custom delimiter."""
    sm = FencedFreeformStateMachine(identifier="json")
    input_sequence = "```json\n{\"key\": \"value\"}\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_fenced_freeform_missing_open_delimiter():
    """Test rejection when open delimiter is missing."""
    sm = FencedFreeformStateMachine()
    input_sequence = "Some freeform text\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_fenced_freeform_missing_close_delimiter():
    """Test rejection when close delimiter is missing."""
    sm = FencedFreeformStateMachine()
    input_sequence = "```\nSome freeform text"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_fenced_freeform_partial_delimiter():
    """Test rejection when delimiter is partially provided."""
    sm = FencedFreeformStateMachine()
    input_sequence = "``\nSome freeform text\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_fenced_freeform_respects_char_min():
    """Test that char_min is respected."""
    sm = FencedFreeformStateMachine(char_min=10)
    input_sequence = "```\nshort\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)

    input_sequence_valid = "```\nlong enough text\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence_valid)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_fenced_freeform_respects_char_max():
    """Test that char_max is respected."""
    sm = FencedFreeformStateMachine(char_max=10)
    input_sequence = "```\nthis text is too long\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)

    input_sequence_valid = "```\nshort\n```"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence_valid)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_fenced_freeform_invalid_continuations():
    """Test invalid continuations within fenced freeform."""
    sm = FencedFreeformStateMachine()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "```\nSome text")
    assert len(steppers) > 0
    for stepper in steppers:
        if stepper.target_state == 2:
            invalid_continuations = stepper.get_invalid_continuations()
            assert "\n```" in invalid_continuations

def test_fenced_freeform_buffer_length():
    """Test buffer_length constraint."""
    sm = FencedFreeformStateMachine(buffer_length=5)
    steppers = sm.get_steppers()
    assert not any(stepper.should_start_step("```\n") for stepper in steppers)

    steppers = sm.advance_all_basic(steppers, "12345")
    assert any(stepper.should_start_step("```\n") for stepper in steppers)
