# Proxy Structuring Engine (PSE)

<p align="center">
  <img src="logo.png" alt="PSE Logo" height="300"/>
</p>

<p align="center">
  <strong>Bringing Order to Chaos: Efficient schema-guided sampling for LLMs</strong>
</p>

<p align="center">
  <img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Python CI"/>
  <img src="https://img.shields.io/badge/coverage-90%25-brightgreen" alt="Coverage"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"/>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#license">License</a>
</p>

## Overview

The Proxy Structuring Engine (PSE) works in tandem with LLMs to ensure generated output conforms to a defined JSON schema.

The PSE enables error free custom tool calling, complex multi-step reasoning, and unlocks new creative possibilities for AI applications.

PSE achieves this through a novel schema-guided sampling approach, leveraging a Directed Acyclic Word Graph (DAWG) and finite state machines.

## Get Started

```bash
pip install pse
```

## Key Features

* **JSON Schema Sampling:** Enforces schema constraints while maintaining creativity in model outputs.
* **Enhanced Tool Calling:** Enables precise tool integration by guaranteeing valid JSON output, streamlining workflows and automation.
* **Universal Compatibility:** Works with any LLM that provides logits or log probabilities, both locally and via API.
* **Enhanced Creativity:** Balances structure with quality, generating actionable and creative outputs that meet your schema requirements.
* **Performance Optimized:** Incorporates several optimizations for speed and efficiency, including:
    * **DAWG (Directed Acyclic Word Graph):** Efficiently validates tokens against the schema.
    * **Lazy Evaluation with Logits:** Processes tokens only as needed.
* **Expanded Schema Support:** Supports JSON Schema with plans to expand to other formats (SQL, Cypher, Python, U-DIFF).
* **Direct HuggingFace Integration:** Seamlessly integrates with HuggingFace `transformers`.
* **Comprehensive Unit Testing:** Ensures code reliability with 90% test coverage.
* **Detailed Documentation and Type Hinting:** Improves readability and developer experience.
* **Hooks for Custom Logic:** `start_hook` and `end_hook` callbacks enable custom logic injection.
* **Robust Error Handling:** Facilitates debugging and integration.

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
