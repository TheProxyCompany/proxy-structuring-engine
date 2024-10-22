# Proxy Structuring Engine (PSE)

<p align="center">
  <img src="logo.png" alt="" height="300"/>
</p>

<p align="center">
  <strong>Bringing Order to Chaos: Efficient Schema-Guided Sampling for LLMs</strong>
</p>

<p align="center">
  <!-- Badges -->
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml"><img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
  <a href="#benchmarks">Benchmarks</a>
</p>

## Overview

The Proxy Structuring Engine (PSE) works in tandem with Large Language Models (LLMs) to ensure generated outputs adhere to predefined schemas without compromising creativity, speed, or context. This enables error-free custom tool calling, complex multi-step reasoning, and unlocks new creative possibilities for AI applications.

PSE achieves this through a novel schema-guided sampling approach, leveraging Directed Acyclic Word Graphs (DAWGs) and finite state machines to enforce constraints during text generation.

## Installation

PSE supports multiple backends such as **PyTorch**, **JAX**, and **MLX** for maximum flexibility and performance.

To install the base package:

```bash
pip install pse
```

To install with optional backend support, use one of the following:

```bash
pip install pse[torch]  # PyTorch support
pip install pse[jax]    # JAX support
pip install pse[mlx]    # MLX support
pip install pse[all]    # All optional features
```

## Features

- 🚀 **Multi-Backend Support**: Compatible with **PyTorch**, **JAX**, and **MLX** backends.
- 🛠 **Schema-Guided Sampling**: Enforces JSON schema constraints during text generation.
- ⚡ **High Performance**: Minimal overhead ensures fast generation speeds.
- 🎨 **Maintains Creativity**: Preserves model creativity while enforcing structure.
- 🤖 **Easy Integration**: Seamlessly integrates with Hugging Face Transformers.
- 📚 **Expanded Schema Support**: Supports JSON Schema with plans to expand to other formats (SQL, Cypher, Python, U-DIFF).
- ✅ **Comprehensive Testing**: Ensures code reliability with 90% test coverage.
- 🔍 **Detailed Documentation**: Improves readability and developer experience.
- 🧩 **Customizable Hooks**: `start_hook` and `end_hook` enable custom logic injection.
- 🔄 **Robust Error Handling**: Facilitates debugging and integration.

## Benchmarks

The Proxy Structuring Engine consistently outperforms traditional sampling methods in both speed and accuracy:

*(Benchmarks will be added soon.)*

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

PSE builds upon the groundwork laid by [LLM Structured Output](https://github.com/otriscon/llm-structured-output) and utilizes [lexpy](https://github.com/aosingh/lexpy) for efficient lexicon analysis.

---

<p align="center">
  Made with care ❤️ by The Proxy Company
</p>

<p align="center">
  <a href="https://x.com/whatisproxy">Twitter</a> •
  <a href="https://www.what-is-proxy.com">Website</a> •
  <a href="mailto:contact@what-is-proxy.com">Contact</a>
</p>
