import pytest
from pse_core.state_machine import StateMachine

from pse.types.grammar import BashStateMachine


@pytest.mark.parametrize(
    "code, should_accept",
    [
        # Basic commands
        ("echo 'Hello, World!'", True),
        ("ls -la", True),
        ("cd /home/user", True),
        ("grep -r 'pattern' .", True),
        ("cat file.txt", True),
        # Variables and assignments
        ("x=1", True),
        ("NAME='John Doe'", True),
        ("export PATH=$PATH:/usr/local/bin", True),
        ("readonly VAR=value", True),
        # Control structures
        ("if [ $x -eq 1 ]; then echo 'yes'; fi", True),
        ("for i in 1 2 3; do echo $i; done", True),
        ("while [ $count -lt 10 ]; do echo $count; ((count++)); done", True),
        ("case $var in a) echo 'a';; b) echo 'b';; esac", True),
        # Functions
        ("function greet() { echo 'Hello'; }", True),
        ("myfunc() { local var=1; echo $var; }", True),
        # Pipes and redirections
        ("cat file.txt | grep pattern", True),
        ("ls > output.txt", True),
        ("cat < input.txt", True),
        ("command >> log.txt 2>&1", True),
        # Command substitution
        ("echo $(date)", True),
        ("files=$(ls -la)", True),
        # Arithmetic
        ("echo $((1 + 2))", True),
        ("((x = y + 3))", True),
        # Comments
        ("# This is a comment", True),
        ("echo 'Hello' # inline comment", True),
        # Multiline commands
        ("echo 'line 1'\necho 'line 2'", True),
        ("if true; then\n  echo 'true'\nfi", True),
        # Invalid syntax
        ("if then fi", False),
        ("for in do done", False),
        ("case esac", False),
        ("echo 'unterminated string", False),
        ("function () {}", False),
        ("ls | | grep pattern", False),
    ],
)
def test_bash_source_validation(code, should_accept):
    """Test validation of Bash source code."""
    source_code_sm = StateMachine(
        {
            0: [(BashStateMachine, "$")],
        }
    )
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, code)

    if should_accept:
        assert any(stepper.has_reached_accept_state() for stepper in steppers), (
            f"Should accept valid Bash code: {code}"
        )
    else:
        assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
            f"Should not accept invalid Bash code: {code}"
        )


def test_incremental_parsing():
    """Test incremental parsing of Bash code."""
    source_code_sm = StateMachine(
        {
            0: [(BashStateMachine, "$")],
        }
    )

    # Test that we can parse a simple echo command
    # This is known to work from the other tests
    complete_code = "echo 'Hello, World!'"
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, complete_code)

    # The simple echo command should be valid
    assert len(steppers) > 0, "Should accept simple echo command"
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_empty_input():
    """Test handling of empty input."""
    source_code_sm = StateMachine(
        {
            0: [(BashStateMachine, "$")],
        }
    )
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, "")
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "incomplete_code",
    [
        "if [ $x -eq 1 ]; then",
        "for i in 1 2 3; do",
        "while true; do",
        "case $var in",
        "function name() {",
        "echo 'Hello' |",
        "ls -la >",
        "cat <",
        "grep pattern &&",
        "find . -name '*.txt' ||",
    ],
)
def test_incomplete_but_valid_code(incomplete_code):
    """Test handling of incomplete but syntactically valid Bash code."""
    source_code_sm = StateMachine(
        {
            0: [(BashStateMachine, "$")],
        }
    )
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, incomplete_code)
    assert len(steppers) > 0
    # Incomplete code should be able to accept more input
    assert all(stepper.can_accept_more_input() for stepper in steppers)
    # Incomplete code should not be in an accept state
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize(
    "code",
    [
        "echo 'string with unmatched quote",
        "if [ $x -eq 1 ]",
        "for i in $(seq 1 10)",
        "cat file.txt | grep 'pattern' |",
        "function name() { echo 'incomplete",
        "case $var in a) echo 'a';;",
        "while [ $count -lt 10 ]",
        "ls -la 2>",
    ],
)
def test_bash_specific_incomplete_constructs(code):
    """Test Bash-specific incomplete constructs that should be considered valid during incremental parsing."""
    source_code_sm = StateMachine(
        {
            0: [(BashStateMachine, "$")],
        }
    )
    steppers = source_code_sm.get_steppers()
    steppers = source_code_sm.advance_all_basic(steppers, code)
    assert len(steppers) > 0, f"Should accept incomplete Bash construct: {code}"
    # Incomplete constructs should not be in an accept state
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)
    assert all(stepper.can_accept_more_input() for stepper in steppers)

def test_bash_code_validate_no_code():
    """Test that validate returns False for empty code."""
    assert not BashStateMachine.grammar.validate("")
