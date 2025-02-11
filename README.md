<p align="center">
  <img src="logo.png" alt="" height="300"/>
</p>

<p align="center">
  <strong>Stateful control of Large Language Models</strong>
</p>

<p align="center">
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml"><img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
   <a href="https://pypi.org/project/pse/"><img src="https://badge.fury.io/py/pse.svg" alt="PyPI version"></a>
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

# Proxy Structuring Engine (PSE)

Move beyond the limitations of prompt engineering, regex, and post-processing. The *Proxy Structuring Engine* (PSE) is a system for repurposing a stochastic LLM as a stateful, controllable engine capable of powering complex workflows and applications.

## How it Works

Imagine laying down train tracks for an LLM. The LLM is the powerful engine, but the tracks (the state machine) dictate where it can go. The train tracks are represented as a Hierarchical State Machine (HSM), which is a complex blueprint or flowchart that defines the LLM's behavior.

*   **State Machine as a Blueprint:** The HSM defines the allowed sequences of tokens. Each state represents a point in the generation process (e.g., "inside a JSON object," "parsing an integer," "within a Python function definition"). Transitions between states represent valid steps.

*   **Steppers: Tracking Progress:** A "Stepper" object keeps track of the LLM's current position within the state machine. It remembers the generation history and knows which transitions (and therefore, which tokens) are valid next.

*   **Logit Masking: Enforcing the Rules:** Before each token is sampled, PSE examines the current Stepper's state. It identifies all valid next tokens according to the state machine. It then masks the logits (the LLM's raw output probabilities – numerical scores assigned to each possible next token) of all invalid tokens, setting their probabilities to effectively zero.

*   **Custom Sampling:** The engine provides a method that takes log probabilities and a user defined sampling function (e.g., temperature sampling, top-k sampling). This ensures that the LLM only chooses from the grammatically correct options while allowing for different sampling strategies.

*   **Parallel Exploration:** Many grammars are ambiguous and allow for multiple valid paths. The PSE handles this by maintaining multiple Steppers simultaneously, each exploring a different possibility. It then selects the "best" path based on the LLM's own probabilities and current state.

## Key Features

PSE includes features such as:

*   **Token Healing:** Automatically corrects minor errors in generated tokens, giving partial credit to the valid prefix.

*   **Multi-Token Continuations:** Handles situations where a single element could be expanded into multiple tokens.

*   **Natural Language & Structure Integration:** Enables a "scratchpad" for free-form LLM output before transitioning to structured generation, triggered by user-defined delimiters. The engine waits for the LLM to produce specific delimiters, allowing the LLM to naturally introduce, explain, or reason about its structured output.

*   **Custom Grammars:** Supports not just JSON Schema and Pydantic models, but also any ENBF grammar or even custom grammar definable via a Python API. This includes entire programming languages (like Python), custom data formats, or even the rules of a game.

## Stateful LLMs: The Advantage

By combining state machines, logit masking, and custom sampling, PSE effectively renders LLMs stateful. A stateful LLM is one whose output at any point depends not just on the prompt, but also on the history of its previous outputs, as tracked by a state machine. This unlocks capabilities previously difficult or impossible to achieve:

*   **Complex, Multi-Step Tool Use:** The LLM can now follow a sequence of human-defined steps, where each step depends on the previous ones.

*   **Planning and Reasoning:** The LLM can explore any path through the state machine.

*   **Context Retention:** The state machine can maintain context and enforce constraints over long interactions.

## Technical Details

*   **C++ Core:** The PSE's core logic is implemented in C++ to achieve maximum performance, enabling real-time logit masking and state machine traversal without introducing significant latency.

*   **Custom Sampling:** The PSE allows developers to fine-tune the LLM's creativity by implementing custom sampling strategies, while ensuring that the output remains structurally valid.

*   **Extensible Design:** The system is designed to be modular and extensible, allowing developers to add support for new grammars, state machine types, and LLM frontends.

*   **Model Agnostic:** PSE works with any LLM that exposes its logits (output probabilities) and allows for custom sampling behavior. This includes popular frameworks like Hugging Face Transformers, MLX, and others.

*   **Open Source:** The Proxy Structuring Engine python package is released under the Apache 2.0 license, encouraging community contributions and collaboration. The core C++ engine is distributed as a compiled binary.

*   **Python Bindings:** A user-friendly Python API (built with `nanobind`) provides seamless access to the C++ core.

*   **Zero-Copy Data Sharing:** `nanobind` enables efficient data transfer between Python and C++ without unnecessary copying.

## Applications

PSE is ideally suited for any application where consistent output from an LLM is required, including:

*   **Intelligent Agents:** Building agents that can interact with APIs, databases, and external tools.

*   **Data Extraction and Transformation:** Extracting structured information from unstructured text with guaranteed accuracy.

*   **Interactive Fiction and Games:** Creating dynamic narratives and game worlds with complex rules and state management.

*   **Chatbots and Conversational AI:** Developing chatbots that can understand and respond to user input in a structured manner.

## Performance

### Latency

*   First-Token Latency: Minimal, no preprocessing or index building required
*   Per-Token additional latency: ~0.02s (20ms)
*   Memory: Zero copy tensor ops, minimal memory footprint

### Scaling

*   Linear complexity growth with grammar size
*   Parallel path evaluation without multiplicative overhead

### Use Cases

*   API-driven agents with guaranteed output schemas
*   Complex tool use, supporting high tool number
*   Unstructured text → structured data conversion
*   Syntax-constrained code generation
*   Game state machines with narrative continuity

## Conclusion

The Proxy Structuring Engine empowers developers to build reliable, stateful LLM applications by transforming probabilistic language models into deterministic, controllable engines.

By shifting the focus from post-hoc correction to integrated, real-time steering, the PSE enables a new class of robust, reliable, and stateful LLM applications. It's a powerful tool for anyone who needs more than just unpredictable text from their language models.
