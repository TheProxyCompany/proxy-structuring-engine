from __future__ import annotations

import pytest

from pse.types.base.phrase import PhraseStateMachine
from pse.types.xml.xml_encapsulated import XMLEncapsulatedStateMachine


def test_basic_wrapped_content() -> None:
    """Test recognition of basic wrapped content."""
    inner_sm = PhraseStateMachine("content")
    wrapped_sm = XMLEncapsulatedStateMachine(inner_sm, "div")
    steppers = wrapped_sm.get_steppers()

    input_sequence = "<div>content</div>"
    steppers = wrapped_sm.advance_all_basic(steppers, input_sequence)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_incremental_parsing() -> None:
    """Test incremental parsing of wrapped content."""
    inner_sm = PhraseStateMachine("hello")
    wrapped_sm = XMLEncapsulatedStateMachine(inner_sm, "greeting")
    steppers = wrapped_sm.get_steppers()

    parts = ["<greeting", ">", "hel", "lo", "</greeting", ">"]

    for part in parts:
        steppers = wrapped_sm.advance_all_basic(steppers, part)
        assert len(steppers) > 0

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_nested_wrapped_content() -> None:
    """Test nested XML wrapped content."""
    inner_content = PhraseStateMachine("text")
    inner_wrapped = XMLEncapsulatedStateMachine(inner_content, "span")
    outer_wrapped = XMLEncapsulatedStateMachine(inner_wrapped, "div")

    input_sequence = "<div><span>text</span></div>"
    steppers = outer_wrapped.get_steppers()
    steppers = outer_wrapped.advance_all_basic(steppers, input_sequence)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_min_buffer_length() -> None:
    """Test that min_buffer_length is respected."""
    inner_sm = PhraseStateMachine("content")
    wrapped_sm = XMLEncapsulatedStateMachine(inner_sm, "div", min_buffer_length=10)
    steppers = wrapped_sm.get_steppers()

    # Should not start with opening tag yet
    assert not any(stepper.should_start_step("<div>") for stepper in steppers)

    # Add sufficient buffer
    buffer_content = "x" * 10
    steppers = wrapped_sm.advance_all_basic(steppers, buffer_content)
    assert any(stepper.should_start_step("<div>") for stepper in steppers)


def test_invalid_content() -> None:
    """Test rejection of invalid inner content."""
    inner_sm = PhraseStateMachine("expected")
    wrapped_sm = XMLEncapsulatedStateMachine(inner_sm, "div")
    steppers = wrapped_sm.get_steppers()

    input_sequence = "<div>unexpected</div>"
    steppers = wrapped_sm.advance_all_basic(steppers, input_sequence)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_missing_closing_tag() -> None:
    """Test rejection when closing tag is missing."""
    inner_sm = PhraseStateMachine("content")
    wrapped_sm = XMLEncapsulatedStateMachine(inner_sm, "div")
    steppers = wrapped_sm.get_steppers()

    input_sequence = "<div>content"
    steppers = wrapped_sm.advance_all_basic(steppers, input_sequence)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_mismatched_tags() -> None:
    """Test rejection of mismatched opening and closing tags."""
    inner_sm = PhraseStateMachine("content")
    wrapped_sm = XMLEncapsulatedStateMachine(inner_sm, "div")
    steppers = wrapped_sm.get_steppers()

    input_sequence = "<div>content</span>"
    steppers = wrapped_sm.advance_all_basic(steppers, input_sequence)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "tag_name, content, should_accept",
    [
        ("div", "content", True),
        ("p", "multi\nline\ncontent", True),
        ("custom-tag", "content with spaces", True),
    ],
)
def test_various_content_scenarios(
    tag_name: str, content: str, should_accept: bool
) -> None:
    """Test various content scenarios."""
    inner_sm = PhraseStateMachine(content)
    wrapped_sm = XMLEncapsulatedStateMachine(inner_sm, tag_name)
    steppers = wrapped_sm.get_steppers()

    input_sequence = f"<{tag_name}>{content}</{tag_name}>"
    steppers = wrapped_sm.advance_all_basic(steppers, input_sequence)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
