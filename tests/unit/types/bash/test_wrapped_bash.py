import pytest

from pse.types.base.encapsulated import EncapsulatedStateMachine
from pse.types.grammar import BashStateMachine


def test_basic_bash_block():
    """Test basic Bash code block parsing."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()

    # Test opening delimiter
    steppers = bash_sm.advance_all_basic(steppers, "```bash\n")
    assert len(steppers) > 0
    # Test Bash code
    steppers = bash_sm.advance_all_basic(steppers, "echo 'Hello, World!'")
    assert len(steppers) > 0

    # Test closing delimiter
    steppers = bash_sm.advance_all_basic(steppers, "\n```")
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "code_block",
    [
        "```bash\necho 'Hello, World!'\n```",
        "```bash\nls -la\ngrep pattern file.txt\n```",
        "```bash\nfor i in 1 2 3; do\n  echo $i\ndone\n```",
        "```bash\nif [ -f file.txt ]; then\n  cat file.txt\nelse\n  echo 'File not found'\nfi\n```",
        "```bash\nfunction greet() {\n  echo 'Hello, $1!'\n}\n\ngreet 'World'\n```",
    ],
)
def test_complete_bash_blocks(code_block):
    """Test various complete Bash code blocks."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()
    steppers = bash_sm.advance_all_basic(steppers, code_block)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_custom_delimiters():
    """Test BashStateMachine with custom delimiters."""
    sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=("<bash>", "</bash>")
    )
    steppers = sm.get_steppers()

    code_block = "<bash>echo 'Hello, World!'</bash>"
    steppers = sm.advance_all_basic(steppers, code_block)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_stepper_clone():
    """Test cloning of BashStepper."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()
    steppers = bash_sm.advance_all_basic(steppers, "```bash\necho 'Hello'\n")

    original_stepper = steppers[0]
    cloned_stepper = original_stepper.clone()

    assert original_stepper.get_current_value() == cloned_stepper.get_current_value()
    assert original_stepper is not cloned_stepper


@pytest.mark.parametrize(
    "invalid_block",
    [
        "```bash\nif then fi\n```",
        "```bash\nfor in do done\n```",
        "```bash\ncase esac\n```",
        "```bash\nfunction () {}\n```",
        "```bash\nls | | grep pattern\n```",
    ],
)
def test_invalid_bash_blocks(invalid_block):
    """Test handling of invalid Bash code blocks."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()
    steppers = bash_sm.advance_all_basic(steppers, invalid_block)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_incremental_parsing():
    """Test incremental parsing of Bash code block."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()

    parts = [
        "```bash\n",
        "for i in 1 2 3; do\n",
        "  echo $i\n",
        "done\n",
        "```",
    ]

    for part in parts:
        steppers = bash_sm.advance_all_basic(steppers, part)
        assert len(steppers) > 0

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_can_accept_more_input():
    """Test that the stepper can accept more input."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()
    steppers = bash_sm.advance_all_basic(steppers, "```bash\n")
    assert len(steppers) > 0
    steppers = bash_sm.advance_all_basic(steppers, "echo 'Hello")
    assert len(steppers) > 0
    assert all(stepper.can_accept_more_input() for stepper in steppers)
    steppers = bash_sm.advance_all_basic(steppers, "'")
    assert len(steppers) > 0
    assert all(stepper.can_accept_more_input() for stepper in steppers)
    steppers = bash_sm.advance_all_basic(steppers, "\n```")
    assert len(steppers) > 0
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "code_block",
    [
        "```bash\n#!/bin/bash\n\n# This is a comment\necho 'Hello, World!'\n```",
        "```bash\nVAR='value'\necho $VAR\n```",
        "```bash\nls -la | grep '.txt' | sort\n```",
        "```bash\nif [ -d /tmp ]; then\n  echo 'Directory exists'\nfi\n```",
    ],
)
def test_bash_specific_features(code_block):
    """Test Bash-specific features."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()
    steppers = bash_sm.advance_all_basic(steppers, code_block)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "incomplete_block",
    [
        "```bash\nif [ $x -eq 1 ]; then\n",
        "```bash\nfor i in 1 2 3; do\n",
        "```bash\nwhile true; do\n",
        "```bash\ncase $var in\n",
        "```bash\nfunction name() {\n",
    ],
)
def test_incomplete_bash_blocks(incomplete_block):
    """Test handling of incomplete Bash code blocks."""
    bash_sm = EncapsulatedStateMachine(
        BashStateMachine, delimiters=BashStateMachine.delimiters
    )
    steppers = bash_sm.get_steppers()
    steppers = bash_sm.advance_all_basic(steppers, incomplete_block)
    assert len(steppers) > 0
    assert all(stepper.can_accept_more_input() for stepper in steppers)
