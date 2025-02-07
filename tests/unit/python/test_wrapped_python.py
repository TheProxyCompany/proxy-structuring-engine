import pytest

from pse.lark.grammar.wrapped_grammar import WrappedGrammarStateMachine
from pse.lark.python import Grammar, cached_parse_validation, python_parser


def test_basic_python_block():
    """Test basic Python code block parsing."""
    python_grammar = Grammar(python_parser, cached_parse_validation)
    python_sm = WrappedGrammarStateMachine(
        grammar=python_grammar, delimiters=("```python\n", "\n```")
    )
    steppers = python_sm.get_steppers()

    # Test opening delimiter
    steppers = python_sm.advance_all_basic(steppers, "```python\n")
    assert len(steppers) > 0
    # Test Python code
    steppers = python_sm.advance_all_basic(steppers, "x = 1\n")
    assert len(steppers) > 0

    # Test closing delimiter
    steppers = python_sm.advance_all_basic(steppers, "\n```")
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "code_block",
    [
        "```python\nprint('Hello')\n\n```",
        "```python\nx = 1\ny = 2\nprint(x + y)\n\n```",
        "```python\ndef test():\n    return True\n\n```",
        "```python\nclass Test:\n    pass\n\n```",
    ],
)
def test_complete_python_blocks(code_block):
    """Test various complete Python code blocks."""
    python_grammar = Grammar(python_parser, cached_parse_validation)
    python_sm = WrappedGrammarStateMachine(
        grammar=python_grammar, delimiters=("```python\n", "\n```")
    )
    steppers = python_sm.get_steppers()
    steppers = python_sm.advance_all_basic(steppers, code_block)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_custom_delimiters():
    """Test PythonStateMachine with custom delimiters."""
    python_grammar = Grammar(python_parser, cached_parse_validation)
    sm = WrappedGrammarStateMachine(
        grammar=python_grammar, delimiters=("<python>", "</python>")
    )
    steppers = sm.get_steppers()

    code_block = "<python>x = 1</python>"
    steppers = sm.advance_all_basic(steppers, code_block)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_stepper_clone():
    """Test cloning of PythonStepper."""
    python_grammar = Grammar(python_parser, cached_parse_validation)
    python_sm = WrappedGrammarStateMachine(
        grammar=python_grammar, delimiters=("```python\n", "\n```")
    )
    steppers = python_sm.get_steppers()
    steppers = python_sm.advance_all_basic(steppers, "```python\nx = 1\n")

    original_stepper = steppers[0]
    cloned_stepper = original_stepper.clone()

    assert original_stepper.get_current_value() == cloned_stepper.get_current_value()
    assert original_stepper is not cloned_stepper


@pytest.mark.parametrize(
    "invalid_block",
    [
        "```python\ndef invalid syntax\n\n```",
        "```python\nclass:\n\n```",
        "```python\nwhile:\n\n```",
        "```python\nprint('no closing parenthesis'\n\n```",
    ],
)
def test_invalid_python_blocks(invalid_block):
    """Test handling of invalid Python code blocks."""
    python_grammar = Grammar(python_parser, cached_parse_validation)
    python_sm = WrappedGrammarStateMachine(
        grammar=python_grammar, delimiters=("```python\n", "\n```")
    )
    steppers = python_sm.get_steppers()
    steppers = python_sm.advance_all_basic(steppers, invalid_block)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_incremental_parsing():
    """Test incremental parsing of Python code block."""
    python_grammar = Grammar(python_parser, cached_parse_validation)
    python_sm = WrappedGrammarStateMachine(
        grammar=python_grammar, delimiters=("```python\n", "\n```")
    )
    steppers = python_sm.get_steppers()

    parts = [
        "```python\n",
        "def test():\n",
        "    x = 1\n",
        "    return x\n",
        "\n```",
    ]

    for part in parts:
        steppers = python_sm.advance_all_basic(steppers, part)
        assert len(steppers) > 0

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
