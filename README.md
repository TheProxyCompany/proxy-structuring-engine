# Proxy Structuring Engine (PSE)
<p align="center">
  <img src="logo.png" alt="" height="300"/>
</p>

<p align="center">
  <strong>Type-Safe LLMs: Grammar Enforcement at the Speed of Thought</strong>
</p>

<p align="center">
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml"><img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
   <a href="https://pypi.org/project/pse/"><img src="https://badge.fury.io/py/pse.svg" alt="PyPI version"></a>
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

**Turn any LLM into a stateful agent.  Guaranteed structured output, at the speed of thought.**

The Proxy Structuring Engine (PSE) is a powerful, open-source library that brings **stateful, grammar-constrained generation** to *any* Large Language Model (LLM).
Unlike methods that rely on complex regex, fragile post-processing or complex prompting, the PSE enforces structure *during* generation by manipulating logits in real-time.

This results in:
*   **Guaranteed Validity:** Output *always* conforms to a specified schema (JSON Schema, Pydantic models, or custom grammars).
*   **Preserved Creativity:**  PSE guides, but doesn't restrict, the LLM's natural language capabilities.  It works *with* the model's probability distribution, not against it.
*   **Stateful Behavior:**  Enables complex, multi-step interactions and planning by allowing the LLM to drive a hierarchical state machine (HSM).
*   **Universal Compatibility:**  Works with any LLM that exposes logits (including Hugging Face Transformers, MLX, and custom models).
*   **Extensible Grammar Support:**  Define custom grammars using a powerful and intuitive state machine API or leverage pre-built support for JSON Schema, Pydantic, syntaxically valid Python code, and more.
*   **Unmatched Speed:**  C++ core with zero-copy operations delivers microsecond overhead per token.

## Why PSE?

Traditional methods for controlling LLM output (regex, string manipulation, or post-processing) are often brittle, slow, or limit the model's creativity.
PSE takes a fundamentally different approach:
*   **No Post-Processing:**  Structure is enforced *during* generation, eliminating the need for costly and error-prone post-processing.
*   **No Fine-Tuning Required:**  PSE works with *any* off-the-shelf LLM, saving you time and compute resources.
*   **Beyond JSON:**  Support for custom grammars lets you enforce *any* structured output, not just JSON.
    *   Think SQL, syntaxtically correct code, or even a custom DSL.
*   **Stateful Agents:**  PSE's state machine integration unlocks entirely new possibilities for building sophisticated, multi-step LLM applications.

## Key Features:

*   🧠 **Stateful Generation:**  Use the LLM to drive a hierarchical state machine.
*   🚀 **Fast:**  C++ core with zero-copy operations and optimized data structures.
*   🎭 **Flexible:**  Support for:
    *   The full JSON Schema Specification
    *   Pydantic models
    *   Any ENBF grammar (using the `lark` library to parse the grammar)
    *   Design your own state machine using the PSE state machine API.
*   🛡️ **Type Safety:**  Catch schema violations *before* they manifest.
*   🔌 **Framework Agnostic:** Integrates with any LLM stack that exposes logit manipulation (PyTorch, Jax, MLX, etc.).
*   ✨ **Token Healing:**  Gracefully handles partially valid tokens, improving robustness. Enabled by default.

## Quick Start
```bash
pip install pse
```
Or, to develop the pse, install all the needed packages:
```bash
pip install pse[dev]
```

## Advanced Usage

### Custom Grammars

PSE allows you to define custom grammars using a powerful state machine API.  This gives you fine-grained control over the LLM's output.

```python
from pse_core.state_machine import StateMachine
from pse.types.number import NumberStateMachine
from pse.engine import StructuringEngine

# Example:  A simple grammar for arithmetic expressions (e.g., "2 + 3 * 4")
class OperatorStateMachine(StateMachine):
    def __init__(self):
        super().__init__({ 0: [(CharacterStateMachine("+-*/", char_limit=1), "$")] })

class ArithmeticStateMachine(StateMachine):
    def __init__(self):
        super().__init__(
            {
                0: [
                    (NumberStateMachine(), 1)
                ],
                1: [
                    (OperatorStateMachine(), 0),
                    (StateMachine.END_STATE, "$"), # '$' represents a valid end state
                ],
            },
            start_state=0,
            end_states=["$"],
        )

# Now you can use this custom grammar with the StructuringEngine:
engine = StructuringEngine(tokenizer) # Tokenizer is a huggingface tokenizer
engine.configure(ArithmeticStateMachine())
# ... use engine.process_logits() in your generation loop ...
```

### Grammars with `lark`
```python
from lark import Lark
from pse.types.grammar.grammar import GrammarStateMachine, Grammar

# Define your grammar using Lark's EBNF syntax
python_parser = Lark.open_from_package(
    "lark",
    "python.lark",
    ["grammars"],
    start=["file_input"],
    ordered_sets=False,
)

# A function to use to validate the text
def validate(parser, text, strict, start):
    try:
        parser.parse(text)
        return True
    except Exception as e:
        return False

# Create a Grammar Instance.
python_grammar = Grammar(
    name="Python",
    lark_grammar=python_parser,
    validator_function=validate_python_code,
    delimiters=("```python\n", "\n```"),
)

# Create a GrammarStateMachine Instance.
json_state_machine = GrammarStateMachine(json_grammar)
```

### Token Healing

PSE can automatically "heal" partially generated tokens, making generation more robust.  This is enabled by default.


### Multi-Token Continuations
The PSE automatically handles the case where a structured output includes multiple tokens.

### Customization
The `Engine` and `Stepper` objects are subclassable - you can add custom logic by overriding the available methods.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
