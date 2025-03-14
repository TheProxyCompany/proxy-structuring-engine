<p align="center">
  <img src="logo.png" alt="Proxy Structuring Engine" style="object-fit: contain; max-width: 80%;"/>
</p>

<p align="center">
  <strong>Dynamically Constrained Natural Language Generation</strong>
</p>

<p align="center">
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml"><img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/coverage-96%25-brightgreen.svg" alt="Test Coverage"></a>
  <a href="https://pypi.org/project/pse/"><img src="https://badge.fury.io/py/pse.svg" alt="PyPI version"></a>
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://docs.theproxycompany.com/pse/"><img src="https://img.shields.io/badge/docs-available-brightgreen.svg" alt="Documentation"></a>
</p>

# Proxy Structuring Engine (PSE)

The **Proxy Structuring Engine** (PSE) is a system for dynamically constrained natural language generation. It is a novel approach to structured outputs, and achieves state-of-the-art results across several categories (compared to existing methods).

The structuring engine compiles rules and schemas into state machines that filter and sample tokens during generation, guaranteeing structurally valid outputs while preserving the underlying model's capabilities. The **PSE** is framework agnostic, and can be added to most LLM inference pipelines.

## Installation

```bash
pip install pse
```
*or*
```bash
uv pip install pse
```

## "Why should I consider using this library?"

The structuring engine:

-   **Maintains** the real-time state during the LLM's generation, ensuring consistency and context awareness.
-   **Guarantees** output structure (e.g., valid JSON, nested schemas, etc.), eliminating post-processing headaches.
-   **Handles** ambiguity and recursion, enabling complex and nuanced interactions.
-   **Operates** at the token level, striking a balance between flexibility and control.  You get precise structure without sacrificing the model's inherent creativity.
-   **Enforces** structure without impacting the model's creativity.

Move beyond the limitations of prompt engineering, regex, overfit fine-tuning, or index-based masking.

## Use Cases

- **Tool Calling** - Generate precise, validated parameters for function calls
- **API Integration** - Guarantee well-formed outputs for seamless system interoperability
- **Synthetic Data** - Create diverse, schema-conformant datasets for training
- **Structured Output** - Enforce type-safe results for reliable downstream processing
- **Agent Frameworks** - Constrain agent actions and reasoning (see [Proxy Base Agent](https://github.com/TheProxyCompany/proxy-base-agent))

### Feature Comparison

| **Feature**                  | **Prompt Engineering** | **Re-try if Invalid** | **Regex** | **Simple Templating** | **Index Based Masking** | **PSE**       |
|------------------------------|------------------------|-----------------------|-----------|-----------------------|-------------------------|---------------|
| **Guaranteed Structure**     | ❌                     | ❌                    | ❌         | ⚠️ Limited             | ✅                        | ✅           |
| **Handles Recursion**        | ❌                     | ❌                    | ❌         | ❌                    | ✅                       | ✅            |
| **Native token healing**     | ❌                     | ❌                    | ❌         | ❌                    | ❌                       | ✅            |
| **Handles Ambiguity**        | ✅                     | ❌                    | ❌         | ❌                    | ❌                       | ✅            |
| **Flexibility (Content)**    | ✅                     | ✅                    | ❌         | ❌                    | ❌                       | ✅            |
| **Performance**              | ✅                     | ⚠️ Depends on retries  | ❌ Slow    | ✅                    | ✅                       | ✅            |
| **Integration with LLMs**    | ✅                     | ⚠️ Post-processing required  | ⚠️ Post-processing required | ⚠️ Post-processing required | ✅  | ✅  |
| **Extensibility**            | ✅                     | ❌                    | ❌        | ❌                    | ❌                       | ✅             |
| **Stateful**                 | ❌                     | ❌                    | ❌        | ❌                    | ❌                       | ✅             |

___

## Quickstart

Here's how to use PSE to generate a simple JSON response:

```python
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin


# 1. Apply the PSE mixin to your model.  This integrates PSE's functionality.
class PSE_Torch(PSETorchMixin, LlamaForCausalLM):
    pass

# 2. Load your model and tokenizer.
model_path = "meta-llama/Llama-3.2-1B-Instruct"  # Any Hugging Face model will work.
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PSE_Torch.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

# Ensure padding token is set for generation.
model.config.pad_token_id = model.config.eos_token_id[0]
if model.generation_config:
    model.generation_config.pad_token_id = model.config.eos_token_id[0]

# 3. Create the StructuringEngine and configure it with your desired schema.
#    Here, we define a simple JSON schema requiring an "answer" key with a string value.
model.engine = StructuringEngine(tokenizer)
schema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
model.engine.configure(schema)

# 4.  Create your prompt.
prompt = 'Please respond with a JSON object with the key "answer" and the value "Hello, world!"'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# 5. Generate!
#    disable truncation samplers (like top_p) to allow PSE to control the output.
#    PSE needs to see all possible tokens to enforce the schema.
output = model.generate(input_ids, do_sample=True, top_p=None)

# Expected output WITH the PSE:
# {"answer": "Hello, world!"}
#
# Example output WITHOUT the PSE (unpredictable):
# Sure! Here's your answer: { "text": "Hello, world!" } Hope that helps!

print(tokenizer.decode(output[0]))

```
You'll get a guaranteed JSON output: `{"answer": "Hello, world!"}`.

### More Examples

Check out the [examples/](examples/) for more examples and advanced usage:

*   **`quickstart.py`:**
  * An interactive quickstart guide to using PSE with a simple example.
*   **`simple_demo.py`:**
  * Basic generation with simple and advanced schemas.
*   **`thinking_answer.py`:**
  * Demonstrates creating a custom state machine to enforce a "chain-of-thought" reasoning process.
  * This example showcases how to combine different `StateMachine` types to build complex generation workflows.

## Framework-agnostic

PSE works with most modern LLM stacks.

We provide mixins for the Transformers library (PyTorch, Flax, TensorFlow) for easy integration, and the structuring engine exposes both `logits_processor` and `sampler` methods, so you can graft PSE into almost any inference pipeline.

Need to integrate with a custom setup? Just drop in our logit processor and sampler—no workarounds needed.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
The `pse-core` C++ library is distributed as a pre-built package and is not open source.

## Contact

For questions or support, please open an issue on the GitHub repository.
