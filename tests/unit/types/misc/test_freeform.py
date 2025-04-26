from pse.types.misc.freeform import FreeformStateMachine, FreeformStepper


def test_freeform_basic():
    """Test basic functionality of FreeformStateMachine."""
    sm = FreeformStateMachine(end_delimiters=["END"])
    input_sequence = "Some freeform textEND"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    accepted_stepper = next(stepper for stepper in steppers if stepper.has_reached_accept_state())
    assert accepted_stepper.get_raw_value() == "Some freeform textEND"

    # Test token_safe_output removes delimiter
    assert accepted_stepper.get_token_safe_output(lambda x: "".join(chr(i) for i in x)) == "Some freeform text"


def test_freeform_multiple_delimiters():
    """Test FreeformStateMachine with multiple end delimiters."""
    sm = FreeformStateMachine(end_delimiters=["END", "STOP", "FINISH"])

    # Test with first delimiter
    input_sequence = "Text with first delimiterEND"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test with second delimiter
    input_sequence = "Text with second delimiterSTOP"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test with third delimiter
    input_sequence = "Text with third delimiterFINISH"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_freeform_missing_delimiter():
    """Test that text without an ending delimiter is not accepted."""
    sm = FreeformStateMachine(end_delimiters=["END"])
    input_sequence = "Text without delimiter"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_freeform_partial_delimiter():
    """Test that partial delimiters aren't accepted."""
    sm = FreeformStateMachine(end_delimiters=["END"])
    input_sequence = "Text with partial delimiterEN"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_freeform_respects_char_min():
    """Test that char_min is respected."""
    sm = FreeformStateMachine(end_delimiters=["END"], char_min=10)

    # Test with text shorter than char_min
    input_sequence = "shortEND"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test with text meeting char_min requirement
    input_sequence = "long enough textEND"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_freeform_stepper_initialization():
    """Test the initialization of FreeformStepper."""
    sm = FreeformStateMachine(end_delimiters=["END"])
    stepper = sm.get_new_stepper()

    assert isinstance(stepper, FreeformStepper)
    assert stepper.state_machine == sm
    assert stepper.buffer == ""


def test_freeform_token_safe_output():
    """Test the token_safe_output method."""
    sm = FreeformStateMachine(end_delimiters=["END", "STOP"])

    # Test with first delimiter
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "Some textEND")
    accepted_stepper = next(stepper for stepper in steppers if stepper.has_reached_accept_state())
    assert accepted_stepper.get_token_safe_output(lambda x: "".join(chr(i) for i in x)) == "Some text"

    # Test with second delimiter
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "Other textSTOP")
    accepted_stepper = next(stepper for stepper in steppers if stepper.has_reached_accept_state())
    assert accepted_stepper.get_token_safe_output(lambda x: "".join(chr(i) for i in x)) == "Other text"


def test_freeform_get_raw_value():
    """Test the get_raw_value method."""
    sm = FreeformStateMachine(end_delimiters=["END"])
    input_sequence = "Raw text valueEND"
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, input_sequence)

    accepted_stepper = next(stepper for stepper in steppers if stepper.has_reached_accept_state())
    assert accepted_stepper.get_raw_value() == input_sequence

    # Ensure the raw value includes the delimiter
    assert accepted_stepper.get_raw_value().endswith("END")


def test_freeform_string_representation():
    """Test the string representation of FreeformStateMachine."""
    sm = FreeformStateMachine(end_delimiters=["END"])
    assert str(sm) == "FreeformText"
