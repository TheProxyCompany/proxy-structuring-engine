import pytest
from pse_core.state_machine import StateMachine

from pse.grammar.grammar import GrammarStateMachine
from pse.grammar.python import Grammar, python_parser, validate_python_code


@pytest.mark.parametrize(
    "code, should_accept",
    [
        ("x = 1", True),
        ("def foo():\n    pass", True),
        ("class Test:\n    pass", True),
        ("print('Hello, World!')", True),
        ("if True:\n    print('test')", True),
        # Test incomplete code
        ("1 + ", False),
        ("def", False),
        ("for i in", False),
        ("x =", False),
        ("", False),
        ("class:", False),
        ("x y z", False),
        # Test expressions
        ("1 + 2", True),
        ("len([1, 2, 3])", True),
        ("'test'.upper()", True),
        # Test multiline code
        ("x = 1\ny = 2\nz = x + y", True),
        ("def test():\n    x = 1\n    return x", True),
        # Test comments
        ("# This is a comment", True),
        ("x = 1  # inline comment", True),
        ("'''docstring'''", True),
        # Test complex Python features
        ("lambda x: x * 2", True),
        ("try:\n    x()\nexcept:\n    pass", True),
        ("with open('file') as f:\n    pass", True),
        ("[x for x in range(10)]", True),
        # Test invalid syntax
        ("def def", False),
        ("class class", False),
        ("return return", False),
        ("import import", False),
    ],
)
def test_python_source_validation(code, should_accept):
    """Test validation of Python source code."""
    python_grammar = Grammar(
        name="Python",
        lark_grammar=python_parser,
        validator_function=validate_python_code,
    )
    source_code_sm = StateMachine(
        {
            0: [(GrammarStateMachine(python_grammar), "$")],
        }
    )
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, code)

    if should_accept:
        assert any(stepper.has_reached_accept_state() for stepper in steppers), (
            f"Should accept valid Python code: {code}"
        )
    else:
        assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
            f"Should not accept invalid Python code: {code}"
        )


def test_incremental_parsing():
    """Test incremental parsing of Python code."""
    python_grammar = Grammar(
        name="Python",
        lark_grammar=python_parser,
        validator_function=validate_python_code,
    )
    source_code_sm = StateMachine(
        {
            0: [(GrammarStateMachine(python_grammar), "$")],
        }
    )
    steppers = source_code_sm.get_steppers()

    # Test valid incremental input
    code_parts = [
        "def test",
        "def",
        " ",
        "(",
        ")",
        ":",
        "\n",
        "    ",
        "x",
        " ",
        "=",
        " ",
        "1",
        "\n",
        "    ",
        "return",
        " ",
    ]

    for part in code_parts:
        steppers = source_code_sm.advance_all_basic(steppers, part)
        assert len(steppers) > 0, f"Should accept partial input: {part}"

    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_empty_input():
    """Test handling of empty input."""
    python_grammar = Grammar(
        name="Python",
        lark_grammar=python_parser,
        validator_function=validate_python_code,
    )
    source_code_sm = StateMachine(
        {
            0: [(GrammarStateMachine(python_grammar), "$")],
        }
    )
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, "")
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "incomplete_code",
    [
        "if x == 1:\n    ",
        "def test():\n    x = ",
        "class MyClass:\n    def ",
        "try:\n    x = 1\nexcept ",
        "with open('file') as ",
        "def foo():",
        "if True:",
        "for x in range(10):",
    ],
)
def test_incomplete_but_valid_code(incomplete_code):
    """Test handling of incomplete but syntactically valid code."""
    python_grammar = Grammar(
        name="Python",
        lark_grammar=python_parser,
        validator_function=validate_python_code,
    )
    source_code_sm = StateMachine(
        {
            0: [(GrammarStateMachine(python_grammar), "$")],
        }
    )
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, incomplete_code)
    assert len(steppers) == 1
    assert all(stepper.can_accept_more_input() for stepper in steppers)
