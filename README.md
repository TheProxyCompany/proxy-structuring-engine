<p align="center">
  <img src="logo.png" alt="Proxy Structuring Engine" style="object-fit: contain; max-width: 80%;"/>
</p>

<p align="center">
  <strong>Steer Your LLM: Stateful Control of Large Language Models</strong>
</p>

<p align="center">
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml"><img src="https://github.com/TheProxyCompany/proxy-structuring-engine/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
   <a href="https://pypi.org/project/pse/"><img src="https://badge.fury.io/py/pse.svg" alt="PyPI version"></a>
  <a href="https://github.com/TheProxyCompany/proxy-structuring-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

# Proxy Structuring Engine (PSE)

LLMs generate words like a firehose with no nozzle—powerful, yet chaotic.

The PSE isn't a filter, but a valve; turning a stochastic LLM into a **stateful** engine capable of driving complex interactions.

> The PSE allows the model to "color within the lines".

## Installation

```bash
pip install pse
```
*or, for those in the know:*
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

Here's a quickstart example using the PSE with a simple schema:
```python
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin


# 1. Apply the PSE mixin to your model
class PSE_Torch(PSETorchMixin, LlamaForCausalLM):
    pass
# 2. Load your model and tokenizer.
model_path = "meta-llama/Llama-3.2-1B-Instruct"  # any model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PSE_Torch.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
# Ensure padding token is set for generation
model.config.pad_token_id = model.config.eos_token_id[0]
if model.generation_config:
    model.generation_config.pad_token_id = model.config.eos_token_id[0]
# 3. Create the StructuringEngine and configure it with your schema
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
output = model.generate(input_ids, do_sample=True, top_p=None) # disable truncation samplers like top_p
# Example output without the PSE:
# Sure! Here's your answer: { "text": "Hello, world!" } Hope that helps!
#
# Example output with the PSE:
# {"answer": "Hello, world!"}
print(tokenizer.decode(output[0]))
```

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

PSE works with most modern LLM stacks. We provide mixins for the Transformers library (PyTorch, Flax, TensorFlow) for easy integration, and the structuring engine exposes both `logits_processor` and `sampler` methods, so you can graft PSE into almost any inference pipeline. Need to integrate with a custom setup? Just drop in our logit processor and sampler—no workarounds needed.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
The `pse-core` C++ library is distributed as a pre-built package.
Source code availability for pse-core will be determined at a later date.

## Contact

For questions or support, please open an issue on the GitHub repository.
