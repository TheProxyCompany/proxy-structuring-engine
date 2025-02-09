import pytest

from pse.types.base.chain import ChainStateMachine
from pse.types.xml.xml_tag import XMLTagStateMachine


def test_basic_tag_initialization() -> None:
    """Test basic initialization of XMLTagStateMachine."""
    tag_machine = XMLTagStateMachine("div")
    assert isinstance(tag_machine, ChainStateMachine)
    steppers = tag_machine.get_steppers()
    assert len(steppers) == 1
    assert not steppers[0].has_reached_accept_state()


def test_closing_tag_initialization() -> None:
    """Test initialization of closing XMLTagStateMachine."""
    tag_machine = XMLTagStateMachine("div", closing_tag=True)
    steppers = tag_machine.get_steppers()
    assert len(steppers) == 1
    assert not steppers[0].has_reached_accept_state()


def test_basic_tag_recognition() -> None:
    """Test recognition of a basic XML tag."""
    tag_machine = XMLTagStateMachine("div")
    steppers = tag_machine.get_steppers()
    steppers = tag_machine.advance_all_basic(steppers, "<div>")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


def test_closing_tag_recognition() -> None:
    """Test recognition of a closing XML tag."""
    tag_machine = XMLTagStateMachine("div", closing_tag=True)
    steppers = tag_machine.get_steppers()
    steppers = tag_machine.advance_all_basic(steppers, "</div>")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


def test_partial_tag_recognition() -> None:
    """Test partial tag recognition behavior."""
    tag_machine = XMLTagStateMachine("div")
    steppers = tag_machine.get_steppers()
    steppers = tag_machine.advance_all_basic(steppers, "<d")
    assert len(steppers) == 1
    assert not steppers[0].has_reached_accept_state()


@pytest.mark.parametrize(
    "tag_name, input_text, should_accept",
    [
        ("div", "<div>", True),
        ("span", "<div>", False),
        ("p", "<P>", False),  # Case sensitivity test
        ("input", "<input>", True),
        ("br", "<br/>", False),  # Doesn't handle self-closing tags
    ],
)
def test_various_tag_scenarios(
    tag_name: str, input_text: str, should_accept: bool
) -> None:
    """Test various tag scenarios including invalid ones."""
    machine = XMLTagStateMachine(tag_name)
    steppers = machine.get_steppers()
    steppers = machine.advance_all_basic(steppers, input_text)

    if should_accept:
        assert len(steppers) == 1
        assert steppers[0].has_reached_accept_state()
    else:
        assert not any(s.has_reached_accept_state() for s in steppers)


def test_empty_tag_name() -> None:
    """Test that empty tag names are not allowed."""
    with pytest.raises(ValueError):
        XMLTagStateMachine("")


def test_whitespace_handling() -> None:
    """Test that whitespace is not allowed within tags."""
    tag_machine = XMLTagStateMachine("div")
    steppers = tag_machine.get_steppers()
    steppers = tag_machine.advance_all_basic(steppers, "< div>")
    assert not any(s.has_reached_accept_state() for s in steppers)
