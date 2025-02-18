<p align="center">
  <img src="logo.png" alt="Proxy Structuring Engine Logo" height="300"/>
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

LLMs are incredibly powerful, but their output is often unpredictable. Think of a firehose of tokens, with no guarantees about the structure or content.
The *Proxy Structuring Engine* (PSE) repurposes a stochastic LLM as a **stateful** engine capable of powering complex interactions.

> The PSE keeps the model free to generate creative language while ensuring it stays within the paths you've deemed valid.

____

## Installation
```bash
pip install pse
```
*or, for those with taste:*
```bash
uv pip install pse
```

## "Why should I consider using this library?"

The structuring engine:
- **Maintains** the real-time state during the LLM's generation,
- **Guarantees** output structure (e.g., valid syntax, nested schemas, etc.),
- **Handles** ambiguity and recursion,
- **Operates** at the token level, striking a balance between flexibility and control,
- **Enforces** structure without effecting the model's creativity.

Move beyond the limitations of prompt engineering, regex, overfit fine-tuning, or index-based masking.

### Feature Comparison
| Feature                      | Prompt Engineering | Re-try if Invalid | Regex      | Simple Templating | Index Based Masking | PSE            |
|------------------------------|--------------------|-------------------|------------|-------------------|----------------------|----------------|
| Guaranteed Structure         | ❌                 | ❌                | ❌         | ⚠️ (limited)     | ✅                    | ✅             |
| Handles Recursion            | ❌                 | ❌                | ❌         | ❌                | ✅                    | ✅ (by nature handles nested structures)           |
| Handles Partial Tokens       | ❌                 | ❌                | ❌         | ❌                | ❌                    | ✅ (via token healing & a recursive descent algorithm)            |
| Handles Ambiguity            | ✅                 | ❌                | ❌         | ❌                | ❌                    | ✅ (via branching & non-deterministic state exploration) |
| Flexibility (Content)        | ✅                 | ✅                | ❌         | ❌                | ❌                    | ✅ (within structural constraints) |
| Performance                  | ✅                 | ⚠️ (depends on retries) | ❌ (can be slow) | ✅                | ✅                    | ✅ (optimized C++ core) |
| Integration with LLMs        | ✅                 | ⚠️ (requires post-proc) | ⚠️ (requires post-proc) | ⚠️ (requires post-proc) | ❌                    | ✅ (via mixins & inference hooks) |
| Extensibility                | ✅                 | ❌                | ❌         | ❌                | ❌                    | ✅ (custom StateMachine subclasses) |
| Stateful                     | ❌                 | ❌                | ❌         | ❌                | ❌                    | ✅             |

___

## Framework-agnostic
PSE works with most modern LLM stacks. We provide mixins for the Transformers library (PyTorch, Flax, TensorFlow) for easy integration, and the structuring engine exposes both `logits_processor` and `sampler` methods, so you can graft PSE into almost any inference pipeline.

Need to integrate with a custom setup?
No need to contort the framework - just wire our logit processing and sampler into your generation loop.

PSE natively supports PyTorch, TensorFlow, JAX, and MLX.
The `pse-core` package, containing the optimized C++ engine, is shipped as a pre-built binary.

## Examples & Quickstart

Check out the [examples/](examples/) for quickstart examples and advanced usage:

*   **`quickstart.py`:**
  * A quickstart guide to using PSE with a simple example.
*   **`simple_demo.py`:**
  * Basic generation with simple and advanced schemas.
*   **`thinking_answer.py`:**
  * Demonstrates creating a custom state machine to enforce a "chain-of-thought" reasoning process.
  * This example showcases how to combine different `StateMachine` types to build complex generation workflows.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
The `pse-core` C++ library is distributed as a pre-built package. Its source code is not currently available.

## Contact

For questions or support, please open an issue on the GitHub repository.
