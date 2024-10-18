# Proxy Structuring Engine (PSE)

<p align="center">
  <img src="path_to_your_logo.png" alt="PSE Logo" width="200"/>
</p>

<p align="center">
  <strong>Bringing Order to Chaos: Transform Any LLM into a Powerful Action Model</strong>
</p>

<p align="center">
  <a href="#key-features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

## Overview

The Proxy Structuring Engine (PSE) enhances LLM applications by ensuring generated output conforms to a defined JSON schema.

This technology enables exact tool calling, complex processing, and unlocks new creative possibilities for AI applications.

PSE achieves this through a novel schema-guided sampling approach, leveraging a Directed Acyclic Word Graph (DAWG) and optimized finite state machines.

## Key Features

* **JSON Schema Sampling:** Enforces schema constraints while maintaining creativity in model outputs.
* **Enhanced Tool Calling:** Enables precise tool integration by guaranteeing valid JSON output, streamlining workflows and automation.
* **Universal Compatibility:** Works with any LLM that provides logits or log probabilities, both locally and via API.
* **Enhanced Creativity:** Balances structure with innovation, generating actionable and creative outputs that meet your schema requirements.
* **Performance Optimized:** Incorporates several optimizations for speed and efficiency, including:
    * **DAWG (Directed Acyclic Word Graph):** Efficiently validates tokens against the schema.
    * **Lazy Evaluation with Logits:** Processes tokens only as needed.
    * **Multi-Character Advancement:** Faster generation by advancing multiple tokens simultaneously.
    * **Optimized Data Structures:** Reduced memory footprint.
* **Expanded Schema Support:** Supports JSON Schema with plans to expand to other formats (SQL, Cypher, Python, U-DIFF).
* **Direct HuggingFace Integration:** Seamlessly integrates with HuggingFace `transformers`.
* **Comprehensive Unit Testing:** Ensures code reliability with 90% test coverage.
* **Detailed Documentation and Type Hinting:** Improves readability and developer experience.
* **Hooks for Custom Logic:** `start_hook` and `end_hook` callbacks enable custom logic injection.
* **Robust Error Handling:** Facilitates debugging and integration.

## Installation

```bash
pip install pse
```

## How It Works

PSE employs a novel schema-guided sampling technique, leveraging a Directed Acyclic Word Graph (DAWG) and optimized finite state machines. This approach enables:

- Efficient token validation against the schema
- Preservation of model creativity
- Optimized performance through lazy evaluation and multi-character advancement

## Benchmarks

PSE consistently outperforms traditional sampling methods in both speed and accuracy:

// add benchmarks here //

## License

PSE is released under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

PSE builds upon the groundwork laid by [LLM Structured Output](https://github.com/otriscon/llm-structured-output) and utilizes [lexpy](https://github.com/aosingh/lexpy) for efficient lexicon analysis.

---

<p align="center">
  Developed with ❤️ by the Proxy team :)
</p>

<p align="center">
  <a href="https://x.com/whatisproxy">Twitter</a> •
  <a href="https://www.what-is-proxy.com">Website</a> •
  <a href="mailto:contact@what-is-proxy.com">Contact</a>
</p>
