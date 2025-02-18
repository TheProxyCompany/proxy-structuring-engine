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

LLMs generate words like a firehose with no nozzle—powerful, but chaotic. PSE isn’t just a filter, it’s the valve—controlling flow, enforcing structure, and keeping creativity intact.
The *Proxy Structuring Engine* (PSE) repurposes a stochastic LLM as a **stateful** engine capable of powering complex interactions.

> The PSE keeps the model free to generate creative language while ensuring it stays within the paths you've deemed valid.

____

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
| **Guaranteed Structure**     | ❌                     | ❌                    | ❌        | ⚠️ Limited            | ✅                       | ✅            |
| **Handles Recursion**        | ❌                     | ❌                    | ❌        | ❌                    | ✅                       | ✅            |
| **Handles Partial Tokens**   | ❌                     | ❌                    | ❌        | ❌                    | ❌                       | ✅            |
| **Handles Ambiguity**        | ✅                     | ❌                    | ❌        | ❌                    | ❌                       | ✅            |
| **Flexibility (Content)**    | ✅                     | ✅                    | ❌        | ❌                    | ❌                       | ✅            |
| **Performance**              | ✅                     | ⚠️ Depends on retries | ❌ Slow   | ✅                    | ✅                       | ✅            |
| **Integration with LLMs**    | ✅                     | ⚠️ Post-proc required | ⚠️ Post-proc required | ⚠️ Post-proc required | ❌                       | ✅            |
| **Extensibility**            | ✅                     | ❌                    | ❌        | ❌                    | ❌                       | ✅            |
| **Stateful**                 | ❌                     | ❌                    | ❌        | ❌                    | ❌                       | ✅            |

___

## Framework-agnostic
PSE works with most modern LLM stacks. We provide mixins for the Transformers library (PyTorch, Flax, TensorFlow) for easy integration, and the structuring engine exposes both `logits_processor` and `sampler` methods, so you can graft PSE into almost any inference pipeline. Need to integrate with a custom setup? Just drop in our logit processor and sampler—no workarounds needed.

## Examples & Quickstart

**Quickstart:**

Here's a quickstart example using the PSE with a simple schema:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin

# 2. Apply the PSE mixin to your model
class PSE_Torch(PSETorchMixin, AutoModelForCausalLM):
    pass

model_path = "meta-llama/Llama-3.2-1B-Instruct" # any model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PSE_Torch.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
# 3. Create the StructuringEngine
model.engine = StructuringEngine(tokenizer)
# configure it with your schema
schema = {
    "type": "object",
    "properties": { "answer": { "type": "string" } },
    "required": ["answer"],
}
model.engine.configure(schema)

# 4.  Create your prompt.
prompt = f"""
Please respond with a JSON object with the key "answer" and the value "Hello, world!".
"""
# 5. Generate!
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
# disable truncation samplers like top_p
output = model.generate(input_ids, do_sample=True, max_new_tokens=100, top_p=0)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Check out the [examples/](examples/) for more examples and advanced usage:

*   **`quickstart.py`:**
  * An interactive quickstart guide to using PSE with a simple example.
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
