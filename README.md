# Proxy Structuring Engine (PSE)

<p align="center">
  <img src="logo.png" alt="" height="300"/>
</p>

<p align="center">
  <strong>Stateful Structured Sampling for Large Language Models</strong>
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

PSE achieves this through an advanced hierarchical state machine and non-deterministic schema-guided sampling approach.

## Installation

The Structuring Engine is designed to be used as a logit processor and sampler for any LLM.
To install the base package:

```bash
pip install pse
```

To install the structuring engine for unit testing & local development:
```bash
pip install pse[dev]
pip install -r requirements.txt
```

## Features

- **Platform Agnostic**: Compatible with any backend, just needs logits and scores.
- **Maintains Creativity**: Preserves model creativity while enforcing structure.
- **Easy Integration**: Hugging Face Transformer integration.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with care ❤️ by The Proxy Company
</p>

<p align="center">
  <a href="https://x.com/TheProxyCompany">Twitter</a> •
  <a href="https://www.theproxycompany.com">Website</a> •
  <a href="mailto:contact@theproxycompany.com">Contact</a>
</p>
