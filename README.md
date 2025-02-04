# Proxy Structuring Engine (PSE)

<p align="center">
  <img src="logo.png" alt="" height="300"/>
</p>

<p align="center">
  <strong>Probabilistic Grammar Enforcement for LLMs</strong>
</p>

<p align="center">
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml"><img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

The Proxy Structuring Engine (PSE) enforces structured outputs in Large Language Models through probabilistic state machines.

## Why PSE?

Traditional approaches to structured LLM outputs rely on post-processing, regex, or prayer.

PSE takes a different path:
- **Guarantees**: State machines enforce grammar at the token level
- **Low Overhead**: C++ core with zero-copy tensor operations
- **Model Preserving**: Maintains creative sampling within grammatical bounds
- **Agnostic**: Works with any framework, frontend or inference pipeline that allows for logit manipulation and custom samplers

## Quick Start

```bash
# Core package
pip install pse

# Development setup
pip install pse[dev]
```

## How It Works

PSE sits between your model and its outputs.
The engine keeps track of the current state, and place in the schema.
It steers the model towards the correct tokens, while maintaining the model's original intent and coherence.

## Core Features

- 🎯 **Token-Level Validation**: Catch issues before they manifest
- 🚄 **High Performance**: State tracking with minimal overhead
- 🧬 **Schema Flexibility**: JSON Schema, custom grammars, hybrid approaches
- 🔧 **Easy Integration**: Drop-in solution for existing pipelines

## Technical Details

PSE uses hierarchical state machines to track valid token transitions during sampling. This enables:

- Grammar-guided sampling without performance penalties
- Guaranteed schema conformance at generation time
- Preservation of model creativity within defined bounds
- Seamless integration with existing LLM architectures

### Logit Processor
```python
def process_logits(tokens, logits):
    # tokens is unused by the engine, but (tokens, logits)
    # is the standard interface for logit processors
    return engine.process_logits(tokens, logits)
```

### Sampler
```python
def sample(logprobs):
    # use your favorite sampler
    def _sample_fn(logprobs):
        # your custom sampler
        # ...
        return token

    return engine.sample(logprobs, _sample_fn)
```

## Use Cases

- **Tool Use**: Guaranteed valid function calls
- **Structured Chat**: Format-preserving conversations
- **Data Extraction**: Schema-conformant parsing
- **Creative Bounds**: Constrained creative generation

## Coming Soon

- Extended grammar specifications
- Performance optimizations
- Formal benchmarks

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Made with precision by The Proxy Company</strong>
</p>

<p align="center">
  <a href="https://x.com/TheProxyCompany">Twitter</a> •
  <a href="https://www.theproxycompany.com">Website</a> •
  <a href="mailto:contact@theproxycompany.com">Contact</a>
</p>
