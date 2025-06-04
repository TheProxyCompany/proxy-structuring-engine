<p align="center">
  <img src="logo.png" alt="Proxy Structuring Engine" style="object-fit: contain; max-width: 50%; padding-top: 20px;"/>
</p>

<h2 align="center">
  <strong>Guaranteed structured outputs from any language model.</strong>
</h2>

<p align="center" style="font-size: 1.2em; width: 350px; margin: 0 auto; padding-bottom: 20px;">
  Eliminate 100% of schema violations and state tracking failures in your LLM applications.
</p>

<p align="center">
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml"><img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
  <a href="https://pypi.org/project/pse/"><img src="https://img.shields.io/pypi/v/pse.svg" alt="PyPI Version"></a>
  <a href="https://docs.theproxycompany.com/pse/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

## The Problem: Unreliable LLM Outputs

Raw LLM outputs are structurally unpredictable, breaking systems that require specific formats (API calls, tool use, structured data). Prompting for structure is inconsistent; post-processing is inefficient and often fails on complex or nested formats.

## The Solution: Runtime Structural Enforcement

The **Proxy Structuring Engine (PSE)** provides **guaranteed structural validity** by enforcing constraints *during* LLM generation.

Define your required output structure using Pydantic, JSON Schema, function signatures, or PSE's composable types. PSE compiles this into a high-performance Hierarchical State Machine (HSM). Integrated into the generation loop, the HSM engine guides the LLM, ensuring every generated token conforms to the defined structure.

**Result: Structurally perfect output, generated correctly the first time.**

## Core Capabilities

*   **100% Structural Guarantee:** Eliminate schema violations, parsing errors, and malformed outputs. Enables reliable downstream processing and state management.
*   **Handles Complexity & Recursion:** Reliably generate deeply nested JSON, valid code, or custom recursive formats via the core HSM engine.
*   **Flexible Schema Definition:** Configure instantly using **Pydantic models**, **JSON Schema**, **Python function signatures**, or compose custom structures with `pse.types`.
*   **Robust & Resilient:** Built-in **Token Healing** recovers from minor tokenization artifacts. Principled path selection resolves ambiguity deterministically.
*   **High-Performance C++ Core:** Optimized HSM engine delivers guaranteed structure with minimal latency (~20ms/token overhead). ([Benchmarks](https://github.com/TheProxyCompany/llm-structured-output-benchmarks))
*   **Model & Framework Agnostic:** Integrates with any local LLM stack via standard logits processing (`process_logits`) and sampling (`sample`) hooks. Optional mixins simplify `transformers` integration (PyTorch, TF, JAX).
*   **Advanced Grammar Composition:** Use `pse.types` (`Chain`, `Loop`, `Any`, etc.) to build custom HSMs for bespoke structural requirements beyond standard schemas.

## Installation

```bash
pip install pse
```
or
```bash
uv add pse
```
*(Installs the `pse` Python library and its required `pse-core` dependency. See [Installation Docs](https://docs.theproxycompany.com/pse/getting-started/installation/) for framework extras and setup)*

## Quickstart: Pydantic to Guaranteed JSON

This example demonstrates generating JSON output matching a Pydantic schema using `transformers`.

```python
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from pydantic import BaseModel
from pse import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin # Optional: Mixin for easy HF integration

# 1. Define structure
class UserProfile(BaseModel):
    user_id: int
    username: str
    is_active: bool
    roles: list[str]

# 2. (Optional) Apply PSE mixin to model class
class PSE_Llama(PSETorchMixin, LlamaForCausalLM): pass

# 3. Load model & tokenizer
model_path = "meta-llama/Llama-3.2-1B-Instruct" # Example
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PSE_Llama.from_pretrained( # Or your base model class
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 4. Init & Configure Engine (Mixin attaches as model.engine)
model.engine = StructuringEngine(tokenizer)
model.engine.configure(UserProfile) # Compile structure to HSM

# 5. Create prompt & input IDs
prompt = f"Generate a user profile: ID 999, username 'tester', active true, roles ['qa', 'dev']. Output ONLY the JSON."
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to(model.device)

# 6. Generate (PSE hooks applied via mixin or manually)
output_ids = model.generate(
    input_ids, max_new_tokens=150, do_sample=True,
    # Manual hook example (if not using mixin):
    # logits_processor=[model.engine.process_logits], sampler=model.engine.sample
)

# 7. Decode & Parse (Guaranteed by PSE)
output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
structured_output: UserProfile | None = model.engine.get_structured_output(UserProfile)

print("Raw Output (Guided by PSE):\n", output_text)
print("\nParsed Pydantic Object:\n", structured_output)
# Expected: UserProfile(user_id=999, username='tester', is_active=True, roles=['qa', 'dev'])
```

See the [`examples/`](examples/) directory for more use cases, including JSON Schema and custom `StateMachine` composition.

## Why PSE?

PSE's **runtime HSM enforcement** offers fundamental advantages over other methods:

| Capability                | PSE (Runtime HSM)     | Prompting / Retries   | Regex / Post-Proc.  | Simple Masking      |
| :------------------------ | :-------------------- | :-------------------- | :------------------ | :------------------ |
| **Guaranteed Structure**  | **100%**              | Probabilistic         | Fixes Errors        | Flat Only           |
| **Complex/Nested**        | **Handles Natively**  | Brittle / Fails       | Impractical         | Cannot Handle       |
| **Recursion**             | **Handles Natively**  | No                    | No                  | No                  |
| **Reliability**           | **Production Grade**  | Low                   | Error-prone         | Brittle             |
| **Efficiency**            | **Optimized C++**     | Retries Cost          | Slow                | Fast (Simple)       |
| **Token Healing**         | **Built-in**          | N/A                   | N/A                 | Breaks              |

## Integration

Integrate PSE into your generation loop via two hooks:
1.  `logits_processor=[engine.process_logits]`
2.  `sampler=engine.sample` (wraps your base sampler)

Works with PyTorch, TensorFlow, JAX, MLX, etc. Optional `transformers` mixins simplify integration. *(See [Docs](https://docs.theproxycompany.com/pse/) for details)*

## Foundation for Reliable Agents

PSE provides the structural guarantees required for reliable agentic systems, powering the **[Proxy Base Agent (PBA)](https://github.com/TheProxyCompany/proxy-base-agent)** for dependable state transitions and tool use.

## License & Source Availability

*   **`pse` (Python Library):** Open source under Apache 2.0 ([LICENSE](LICENSE)). Provides the Python interface, schema parsing, and integration logic.
*   **`pse-core` (C++ Engine):** Required dependency. Distributed as a pre-compiled binary. Contains the high-performance HSM execution core. Source code is proprietary (The Proxy Company, Patent Pending).

## Contact & Support

*   **Issues/Bugs:** [GitHub Issues](https://github.com/TheProxyCompany/proxy-structuring-engine/issues)
*   **Commercial Services:** [The Proxy Company Business Services](https://theproxycompany.com/business) (Integration, Custom Development, Support)

## Citation

```bibtex
@software{Wind_Proxy_Structuring_Engine_2025,
  author    = {Wind, Jack},
  title     = {{Proxy Structuring Engine: Guaranteed Structured Output from Language Models via Runtime Hierarchical State Machine Enforcement}},
  year      = {2025}, # Adjust year if needed
  publisher = {The Proxy Company},
  version   = {2025.06.1}, # Update version as needed
  date      = {2025-04-15}, # Update release date
  url       = {https://github.com/TheProxyCompany/proxy-structuring-engine}
}
```
